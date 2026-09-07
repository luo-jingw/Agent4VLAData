"""Time-parameterise our resampled waypoints with Ruckig, the way a controller actually would.

The arc-length labels carry no time. They say where to go and in what order, spaced by how much the
trajectory bends rather than by how long it took -- which is the point, because it lets the policy
spend its fifty points where the geometry is hard. Something downstream still has to decide when.
That is the job this script does, with Ruckig: given per-joint velocity, acceleration and jerk
limits, produce the fastest motion through the waypoints that respects all three.

Two deliberate choices, both worth knowing about before reading the numbers:

  no cloud.  Ruckig's community build hands multi-waypoint problems to a hosted API -- it prints
             "calculate trajectory via cloud API" and sends the waypoints off the machine. These are
             the customer's trajectories, so instead each segment is solved locally, waypoint to
             waypoint, with a non-zero pass-through velocity at the join. Cornering speed is scaled
             by how sharply the path turns there, which is what keeps the arm from stopping dead at
             every point without letting it cut a hard corner at full speed.

  limits from the data.  We have no manufacturer envelope for this G2, so the limits are read off
             the recordings: the velocity ceiling is a high quantile of what the operator actually
             commanded, and acceleration and jerk follow from the same traces. They are honest
             stand-ins, not spec, and every number here scales with them -- swap in real limits and
             re-run before quoting a cycle time.

Outputs a speed-vs-time comparison (recording against the Ruckig-timed waypoints, in configuration
space) and a video retimed onto Ruckig's clock, which is what "segmented speed-up" means here: each
stretch plays at the speed the controller would actually drive it.

  cd tmp/tmp_train_G2
  .venv-ruckig/bin/python ruckig_demo.py --episode 0
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import progress_resample as ar  # noqa: E402
import resample_video as rv  # noqa: E402
from ruckig import InputParameter, Result, Ruckig, Trajectory  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DOFS = 22


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: pathlib.Path = ROOT / "data" / "pick-arc-v4"
    source: pathlib.Path = ROOT / "data" / "new" / "suction-bin-picking"
    out_dir: pathlib.Path = ROOT / "out" / "ruckig"
    episode: int = 0
    control_hz: float = 100.0
    # Fraction of the observed per-joint envelope the controller is allowed to use. Below 1 because
    # the quantile already sits in the tail of what a human teleoperator produced.
    velocity_headroom: float = 1.0
    # Time to reach max velocity, and to reach max acceleration. Both were first set far too
    # conservatively (0.35 / 0.12), and jerk was the binding one: raising it alone took episode 0
    # from 40.7 s to 26.6 s, after which acceleration took over as the limit. Measured, with the
    # corner margin re-tuned at each setting and overshoot held at 1.0x:
    #
    #   jerk 0.12 -> 40.7 s   jerk 0.03, accel 0.35 -> 26.6 s   accel 0.25 -> 23.8 s
    #   jerk 0.06 -> 33.1 s   accel 0.15 -> 21.0 s              accel 0.10 -> 18.7 s
    #
    # The recording is 18.83 s, so the last row matches a human at full tilt. These defaults stop
    # one step short of that: still guesses, and worth leaving headroom until real spec arrives.
    accel_seconds: float = 0.15
    jerk_seconds: float = 0.03
    quantile: float = 0.995
    # Floor under the per-joint ceiling, as a fraction of the median across the joints this task
    # actually exercises. Without it the idle joints set the pace for everyone; see below.
    limit_floor: float = 0.25
    # Margin on the junction-speed bound. The bound below is the acceleration-only one, which assumes
    # the direction change may use the whole segment; jerk limits mean it cannot, so it has to be
    # taken with margin. 0.25 is calibrated by the only criterion that matters here -- the arm stops
    # overshooting. Measured on episode 0: at 1.00 the trajectory walks 7.6x the path length and
    # takes 169 s, at 0.50 still 1.9x and 77 s, at 0.25 exactly 1.0x and 41 s, and below that it is
    # 1.0x but needlessly slow (0.10 -> 51 s). The run prints the ratio so a bad value cannot pass
    # unnoticed.
    corner_margin: float = 0.35
    # Points executed from each chunk before the controller takes a fresh one. The policy emits 50;
    # a receding-horizon controller runs a prefix and replans, so the executed path is spliced from
    # many chunks rather than being one continuous curve. 10 of 50 is a typical ratio.
    replan_points: int = 10
    # chunks   what the policy emits, spliced the way a receding-horizon controller consumes it
    # recorded the raw 30 fps frames as waypoints -- the baseline our resampling is meant to beat
    waypoints_from: str = "chunks"
    # Also run the raw recorded frames through the same controller and put both on the plot. That is
    # the comparison that says what the resampling buys: same limits, same solver, same episode, only
    # the waypoints differ.
    with_baseline: bool = True
    video: bool = True


def limits_from_recordings(source: pathlib.Path, args: Args, episodes: int = 40) -> np.ndarray:
    """Per-joint velocity ceiling, read off what the operator actually commanded.

    A quantile rather than the maximum: single-frame spikes in a teleoperated trace are as often
    sensor noise as intent, and sizing the envelope to one of them would let every later segment run
    at a speed the arm never really moved at.

    The floor matters more than the quantile. This is a right-handed task, so the whole left arm and
    the head roll sit still through every episode -- their observed ceiling is 0.0006 rad/s, and one
    is exactly zero. Ruckig time-synchronises the axes, so handing those numbers over as limits lets
    a joint that is not doing anything dictate the pace of the arm that is: the first run this way
    took 235 s for an 18.8 s episode. A velocity limit is a property of the actuator, not of what the
    operator happened to ask for, so joints the task leaves idle get the floor instead.
    """
    steps = []
    for path in sorted((source / "data" / "chunk-000").glob("*.parquet"))[:episodes]:
        actions = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy())
        steps.append(np.abs(np.diff(actions[:, :DOFS].astype(np.float64), axis=0)) * ar.FPS)
    observed = np.quantile(np.concatenate(steps), args.quantile, axis=0)
    exercised = observed[observed > 1e-2]
    floor = args.limit_floor * float(np.median(exercised)) if len(exercised) else 0.1
    raised = int((observed < floor).sum())
    print(f"  velocity ceiling floored at {floor:.4f} rad/s, raising {raised} of {DOFS} joints "
          f"(idle in this task: left arm and head roll)")
    return np.maximum(observed, floor)


def passthrough_velocities(waypoints: np.ndarray, v_max: np.ndarray, a_max: np.ndarray,
                           margin: float = 0.25) -> np.ndarray:
    """A velocity to carry through each waypoint: along the path, and slow enough to be reachable.

    Ruckig's local solver takes one target at a time, so the join between segments is where the
    motion would otherwise stop dead. Each waypoint therefore gets a velocity along its own local
    path direction -- but the magnitude is the part that decides whether this works at all.

    Sizing it from the velocity ceiling alone does not: the waypoints here are about 0.07 rad apart,
    and asking the arm to pass through at the 1.5 rad/s the joints could sustain leaves it no room to
    turn, so it overshoots each waypoint and comes back. Measured that way the first attempt walked
    thirteen times the path length and took 229 s for an 18.8 s episode.

    So the speed is capped three ways and then made mutually consistent:

      velocity    the fastest travel along this direction that keeps every joint inside its ceiling
      cornering   scaled by how straight the path runs through the waypoint, zero at a reversal
      reachable   a forward and a backward sweep applying v^2 <= v_prev^2 + 2*a*L over each segment,
                  so no waypoint asks for a speed the neighbouring segment cannot build or shed

    which is the standard path-parameterisation bound; Ruckig then supplies the jerk-limited detail
    within each segment. Both ends are pinned to zero -- the arm starts and finishes at rest.
    """
    n = len(waypoints)
    tangent = np.zeros_like(waypoints)
    speed = np.zeros(n)
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

    for i in range(1, n - 1):
        incoming = waypoints[i] - waypoints[i - 1]
        outgoing = waypoints[i + 1] - waypoints[i]
        norm_in, norm_out = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if norm_in < 1e-9 or norm_out < 1e-9:
            continue
        direction = outgoing / norm_out
        tangent[i] = direction
        active = np.abs(direction) > 1e-9
        ceiling = float(np.min(v_max[active] / np.abs(direction[active]))) if active.any() else 0.0

        # Junction speed. Turning by theta at speed v swings the velocity vector by 2*v*sin(theta/2),
        # and the arm only has the time it spends crossing the segment, about L/v, to do it -- so the
        # acceleration a corner demands grows with the SQUARE of the speed. Scaling the ceiling by
        # cos(theta) instead, as the first version did, barely slows anything: a typical 11.9 deg
        # turn here keeps 98% of full speed, which is how a 0.03 rad segment ended up with a 1.75
        # rad/s entry velocity and a trajectory that overshot to 0.27 rad and looped back.
        swing = 2.0 * np.sin(0.5 * np.arccos(np.clip(float(np.dot(incoming / norm_in, direction)), -1.0, 1.0)))
        accel_here = float(np.min(a_max[active] / np.abs(direction[active]))) if active.any() else float(np.min(a_max))
        corner = np.sqrt(accel_here * min(norm_in, norm_out) / swing) if swing > 1e-6 else ceiling
        speed[i] = min(ceiling, float(corner))

    # Scalar acceleration along the path, projected the same way the ceiling is. Taking a plain
    # min over the joints instead would hand the bound to whichever joint is idle -- their limits
    # come from the floor above, not from the work -- and that alone cost a factor of four.
    accel = np.zeros(n)
    for i in range(n):
        active = np.abs(tangent[i]) > 1e-9
        accel[i] = float(np.min(a_max[active] / np.abs(tangent[i][active]))) if active.any() else float(np.min(a_max))
    for i in range(1, n):  # forward: cannot arrive faster than the run-up allows
        rate = min(accel[i - 1], accel[i])
        speed[i] = min(speed[i], float(np.sqrt(speed[i - 1] ** 2 + 2 * rate * lengths[i - 1])))
    for i in range(n - 2, -1, -1):  # backward: cannot leave faster than the run-out can absorb
        rate = min(accel[i], accel[i + 1])
        speed[i] = min(speed[i], float(np.sqrt(speed[i + 1] ** 2 + 2 * rate * lengths[i])))

    # The margin is applied last, to the reconciled speeds, so it scales the waypoints held by the
    # velocity ceiling as well as those held by the corner bound. Applying it to the corner term
    # alone leaves the straight stretches at full speed, and those are exactly where the arm builds
    # up enough momentum to overshoot the next turn.
    return tangent * (speed * margin)[:, None]


def splice_chunks(chunks: np.ndarray, curve: ar.ProgressCurve, horizon: int,
                  replan: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The waypoint stream a receding-horizon controller would actually execute.

    Feeding Ruckig the whole-episode resampling would flatter the pipeline: that curve is one
    continuous chain, computed once with the entire episode in view. What a controller receives is a
    chunk anchored on the observation in front of it, of which it executes a prefix before asking for
    another. Splicing those prefixes is what actually gets driven, seams included, and the seams are
    where a chunked policy goes wrong -- successive chunks are anchored on different observations and
    need not agree about where the arm is heading.

    Returns the waypoints and, for each, the source frame it represents, so the result can be put on
    the recording's clock.
    """
    n_frames = len(chunks)
    take = min(replan, horizon)
    points, frames, seams = [], [], []
    frame = 0
    while frame < n_frames - 1:
        targets = curve.s[frame] + np.arange(1, horizon + 1, dtype=np.float64) * curve.step
        times = ar.invert(curve, np.clip(targets, 0.0, curve.total))
        piece = chunks[frame][:take, :DOFS]
        if points:
            # The seam: how far the fresh chunk's first point sits from where the previous one left
            # the arm. A chunked policy has to be judged on this -- the two chunks were anchored on
            # different observations, and a controller cannot smooth away a real disagreement.
            seams.append(float(np.linalg.norm(piece[0] - points[-1][-1])))
        points.append(piece)
        frames.append(times[:take])
        advance = int(round(float(times[take - 1])))
        frame = advance if advance > frame else frame + 1
    return np.concatenate(points), np.concatenate(frames), np.asarray(seams)


def time_parameterise(waypoints: np.ndarray, v_max: np.ndarray, a_max: np.ndarray,
                      j_max: np.ndarray, control_hz: float, margin: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve segment by segment, locally, and stitch the results into one sampled trajectory.

    Returns the sample times, the sampled joint positions, and the time each waypoint is reached.
    """
    target_velocity = passthrough_velocities(waypoints, v_max, a_max, margin)
    otg = Ruckig(DOFS)
    dt = 1.0 / control_hz

    times, positions, arrivals = [0.0], [waypoints[0].copy()], [0.0]
    clock = 0.0
    position = waypoints[0].astype(np.float64).copy()
    velocity = np.zeros(DOFS)
    acceleration = np.zeros(DOFS)

    for i in range(1, len(waypoints)):
        inp = InputParameter(DOFS)
        inp.current_position, inp.current_velocity, inp.current_acceleration = position, velocity, acceleration
        inp.target_position = waypoints[i]
        inp.target_velocity = target_velocity[i]
        inp.target_acceleration = np.zeros(DOFS)
        inp.max_velocity, inp.max_acceleration, inp.max_jerk = v_max, a_max, j_max

        trajectory = Trajectory(DOFS)
        result = otg.calculate(inp, trajectory)
        if result not in (Result.Working, Result.Finished):
            # A pass-through velocity Ruckig cannot honour: fall back to stopping at this waypoint,
            # which is always feasible, and note it rather than silently dropping the segment.
            inp.target_velocity = np.zeros(DOFS)
            result = otg.calculate(inp, trajectory)
            if result not in (Result.Working, Result.Finished):
                raise RuntimeError(f"ruckig failed at waypoint {i}: {result}")

        steps = max(int(np.ceil(trajectory.duration / dt)), 1)
        for k in range(1, steps + 1):
            offset = min(k * dt, trajectory.duration)
            new_position, new_velocity, new_acceleration = trajectory.at_time(offset)
            times.append(clock + offset)
            positions.append(np.asarray(new_position))
        position, velocity, acceleration = (np.asarray(x) for x in trajectory.at_time(trajectory.duration))
        clock += trajectory.duration
        arrivals.append(clock)

    return np.asarray(times), np.stack(positions), np.asarray(arrivals)


def speed_series(times: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Configuration-space speed |dq/dt|, the L2 norm over all 22 joints."""
    dt = np.diff(times)
    keep = dt > 1e-9
    speed = np.linalg.norm(np.diff(positions, axis=0)[keep], axis=1) / dt[keep]
    return times[1:][keep], speed


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.dataset / "meta" / "arc_resample.json").read_text())
    cfg = ar.config_from_report(report)
    provenance = {e["new_index"]: e for e in report["episodes"]}[args.episode]

    # The recording, for the comparison; and the built episode, whose action_chunk column is what
    # the policy emits and therefore what the controller has to execute.
    recorded = pd.read_parquet(
        args.source / "data" / "chunk-000" / f"episode_{provenance['source_index']:06d}.parquet"
    )
    built = pd.read_parquet(args.dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    states = np.stack(recorded["state"].to_numpy()).astype(np.float64)
    actions = np.stack(recorded["actions"].to_numpy()).astype(np.float64)

    curve = ar.progress_curve(states, actions, cfg)
    chunks = np.stack(built["action_chunk"].to_numpy()).reshape(len(built), cfg.action_horizon, -1)
    if args.waypoints_from == "recorded":
        waypoints = actions[:, :DOFS]
        waypoint_frames = np.arange(len(actions), dtype=np.float64)
        seams = np.zeros(0)
    else:
        waypoints, waypoint_frames, seams = splice_chunks(
            chunks, curve, cfg.action_horizon, args.replan_points)
    print(f"episode {args.episode} (recording {provenance['source_index']}): "
          f"{len(actions)} frames = {len(actions) / ar.FPS:.2f} s")
    if args.waypoints_from == "recorded":
        print(f"  {len(waypoints)} waypoints: the raw recorded frames, as a baseline")
    else:
        print(f"  spliced {len(waypoints)} waypoints from action_chunk, "
              f"{args.replan_points} of {cfg.action_horizon} points executed per chunk")
        print(f"  chunk seams: p50 {np.median(seams):.5f} rad, max {seams.max():.5f} rad "
              f"({len(seams)} replans; waypoint spacing p50 "
              f"{np.median(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)):.5f})")

    v_max = limits_from_recordings(args.source, args) * args.velocity_headroom
    a_max = v_max / args.accel_seconds
    j_max = a_max / args.jerk_seconds
    print(f"limits from the recordings (q{args.quantile:g}): "
          f"v {v_max.min():.3f}-{v_max.max():.3f} rad/s, "
          f"a {a_max.min():.3f}-{a_max.max():.3f} rad/s^2, j {j_max.min():.2f}-{j_max.max():.2f} rad/s^3")

    times, positions, arrivals = time_parameterise(
        waypoints, v_max, a_max, j_max, args.control_hz, args.corner_margin)
    # The trajectory has to hit every waypoint; if it also wanders between them it is overshooting
    # and looping back, which no amount of tuning downstream will fix.
    path_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    travelled = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    print(f"  travelled / waypoint path = {travelled / path_length:.2f}x "
          f"({'no overshoot' if travelled < 1.1 * path_length else 'OVERSHOOTING -- lower --corner-margin'})")
    duration = float(times[-1])
    recorded_duration = len(actions) / ar.FPS
    print(f"ruckig duration {duration:.2f} s vs recording {recorded_duration:.2f} s "
          f"({recorded_duration / duration:.2f}x)")

    rt, rs = speed_series(np.arange(len(actions)) / ar.FPS, actions[:, :DOFS])
    kt, ks = speed_series(times, positions)
    print(f"configuration-space speed |dq/dt|: recording p50 {np.median(rs):.3f} max {rs.max():.3f}; "
          f"ruckig p50 {np.median(ks):.3f} max {ks.max():.3f} rad/s")

    # ---- the speed comparison -------------------------------------------------------------
    panels = [(rt, rs, f"recording as captured, {recorded_duration:.1f} s", "#888"),
              (kt, ks, f"ruckig on {len(waypoints)} resampled action-chunk waypoints, "
                       f"{duration:.1f} s", "#1f6feb")]
    if args.with_baseline and args.waypoints_from == "chunks":
        base_times, base_positions, _ = time_parameterise(
            actions[:, :DOFS], v_max, a_max, j_max, args.control_hz, args.corner_margin)
        bt, bs = speed_series(base_times, base_positions)
        panels.append((bt, bs, f"ruckig on the {len(actions)} raw recorded frames as waypoints, "
                               f"{base_times[-1]:.1f} s", "#cf222e"))
        print(f"  baseline, raw frames as waypoints: {base_times[-1]:.1f} s "
              f"({base_times[-1] / duration:.2f}x our {duration:.1f} s)")

    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 3.4 * len(panels)), sharex=False)
    for axis, (t, s, label, colour) in zip(axes, panels, strict=True):
        axis.plot(t, s, colour, linewidth=0.9)
        axis.set_ylabel("|dq/dt|  (rad/s)")
        axis.set_title(label, fontsize=10, loc="left")
        axis.grid(alpha=0.25)
        axis.set_ylim(0, max(p[1].max() for p in panels) * 1.05)
    # The suction transients, on both clocks, so the event lines up with what it costs.
    frame_to_ruckig = arrivals
    for window in curve.windows:
        axes[0].axvspan(window.onset / ar.FPS, window.end / ar.FPS, color="#d29922", alpha=0.25, lw=0)
        lo = float(np.interp(window.onset, waypoint_frames, arrivals))
        hi = float(np.interp(window.end, waypoint_frames, arrivals))
        axes[1].axvspan(lo, hi, color="#d29922", alpha=0.25, lw=0)
    axes[-1].set_xlabel("time (s)   -- shaded: suction transients")
    figure.suptitle(
        f"configuration-space speed, episode {args.episode}: recording vs Ruckig on the resampled waypoints",
        fontsize=11)
    figure.tight_layout()
    plot_path = args.out_dir / f"speed_ep{args.episode:03d}.png"
    figure.savefig(plot_path, dpi=140)
    plt.close(figure)
    print(f"wrote {plot_path}")

    # ---- the retimed video ----------------------------------------------------------------
    if args.video:
        fps = int(ar.FPS)
        # Ruckig's clock back to source frames: each waypoint knows both, and between them the map
        # is monotone, so the video plays each stretch at the speed the controller would drive it.
        output_times = np.arange(0.0, duration, 1.0 / fps)
        source_frames = np.interp(output_times, arrivals, waypoint_frames)
        wanted = np.clip(np.rint(source_frames).astype(int), 0, len(actions) - 1).tolist()
        video_src = args.source / "videos" / "chunk-000" / "image" / f"episode_{provenance['source_index']:06d}.mp4"
        video_dst = args.out_dir / f"ruckig_ep{args.episode:03d}.mp4"
        rv.write(wanted, video_src, video_dst, fps)
        print(f"wrote {video_dst}  ({len(wanted)} frames, {len(wanted) / fps:.2f} s)")

    summary = {
        "episode": args.episode, "source_index": provenance["source_index"],
        "recorded_frames": len(actions), "recorded_seconds": round(recorded_duration, 3),
        "waypoints_from": args.waypoints_from, "waypoints": len(waypoints),
        "chunk_seam_rad_p50": round(float(np.median(seams)), 6) if len(seams) else None,
        "chunk_seam_rad_max": round(float(seams.max()), 6) if len(seams) else None, "ruckig_seconds": round(duration, 3),
        "speedup": round(recorded_duration / duration, 3),
        "limits": {"quantile": args.quantile, "headroom": args.velocity_headroom,
                   "v_max_rad_s": [round(float(v), 4) for v in v_max],
                   "accel_seconds": args.accel_seconds, "jerk_seconds": args.jerk_seconds},
        "speed_rad_s": {"recording_p50": round(float(np.median(rs)), 4),
                        "recording_max": round(float(rs.max()), 4),
                        "ruckig_p50": round(float(np.median(ks)), 4),
                        "ruckig_max": round(float(ks.max()), 4)},
    }
    (args.out_dir / f"ruckig_ep{args.episode:03d}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir / f'ruckig_ep{args.episode:03d}.json'}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
