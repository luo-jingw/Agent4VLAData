"""Measure the stutter: four ways of turning chunks into commands, on real chunks from the dataset.

Drives `chunk_playback.ChunkFollower` over a whole episode the way a receding-horizon controller
would -- take a chunk, execute part of it, take the next from a later observation -- and reports what
each strategy does to the commanded joint speed.

The number that matters is the tick-to-tick velocity change. A command stream whose speed jumps
between consecutive ticks is one the servo has to absorb, and that is what the operator feels.
Position error against the demonstration is reported alongside so a strategy cannot win by simply
refusing to move.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python chunk_playback_demo.py
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
import tyro  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import progress_resample as ar  # noqa: E402
import chunk_playback as cp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
DOFS = 22


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: pathlib.Path = ROOT / "data" / "pick-arc-v4"
    source: pathlib.Path = ROOT / "data" / "new" / "suction-bin-picking"
    out_dir: pathlib.Path = ROOT / "out" / "playback"
    episode: int = 0
    exec_hz: float = 30.0  # the controller's own rate; 30 matches the recording's frame rate
    replan_points: int = 10  # points consumed before a fresh chunk is taken
    latency_s: float = 0.08  # async inference lag, deliberately not a whole number of points
    v_nominal: float = 0.75
    dt_min: float = 0.06
    blend_s: float = 0.15


def load(args: Args):
    report = json.loads((args.dataset / "meta" / "arc_resample.json").read_text())
    cfg = ar.config_from_report(report)
    provenance = {e["new_index"]: e for e in report["episodes"]}[args.episode]
    recorded = pd.read_parquet(
        args.source / "data" / "chunk-000" / f"episode_{provenance['source_index']:06d}.parquet"
    )
    built = pd.read_parquet(args.dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    states = np.stack(recorded["state"].to_numpy()).astype(np.float64)
    actions = np.stack(recorded["actions"].to_numpy()).astype(np.float64)
    chunks = np.stack(built["action_chunk"].to_numpy()).reshape(len(built), cfg.action_horizon, -1)
    return cfg, states, actions, chunks


def replan_frames(chunks, curve, cfg, replan):
    """The observation frames a receding-horizon controller would actually query."""
    frames, frame = [], 0
    while frame < len(chunks) - 1:
        frames.append(frame)
        targets = curve.s[frame] + np.arange(1, cfg.action_horizon + 1) * curve.step
        times = ar.invert(curve, np.clip(targets, 0.0, curve.total))
        advance = int(round(float(times[min(replan, cfg.action_horizon) - 1])))
        frame = advance if advance > frame else frame + 1
    return frames


def run_follower(chunks, frames, args, limits=None, blend=True):
    """Drive the follower across the whole episode, returning the commanded stream."""
    config = cp.Playback(exec_hz=args.exec_hz, v_nominal=args.v_nominal, dt_min=args.dt_min,
                         blend_s=args.blend_s if blend else 1e-9, limits=limits)
    follower = cp.ChunkFollower(config)
    stream = []
    for order, frame in enumerate(frames):
        points = chunks[frame][:, :DOFS]
        follower.push(points, latency_s=args.latency_s if order else 0.0)
        # How long executing `replan_points` actually takes under the schedule, rather than a fixed
        # guess -- the points are unevenly spaced, so a guess either truncates the chunk or overruns.
        times = cp.schedule(points, args.v_nominal, args.dt_min)
        span = float(times[min(args.replan_points, len(times)) - 1]) - (args.latency_s if order else 0.0)
        for _ in range(max(int(round(span * args.exec_hz)), 1)):
            position, _ = follower.step()
            stream.append(position)
            if follower.exhausted:
                break
    return np.stack(stream)


def naive(chunks, frames, replan):
    """One point per control tick, no interpolation -- what the controller is doing today."""
    return np.concatenate([chunks[f][:replan, :DOFS] for f in frames])


def linear(chunks, frames, args, cfg):
    """Distance-scheduled but straight lines between points: position continuous, velocity not."""
    stream = []
    dt = 1.0 / args.exec_hz
    for order, frame in enumerate(frames):
        points = chunks[frame][:, :DOFS]
        times = cp.schedule(points, args.v_nominal, args.dt_min)
        # Same entry point and same span as the follower, so this isolates the interpolant and does
        # not also measure a seam the other strategies handle.
        start = args.latency_s if order else 0.0
        grid = np.arange(start, float(times[min(args.replan_points, len(times)) - 1]), dt)
        stream.append(np.stack([
            np.array([np.interp(t, times, points[:, d]) for d in range(DOFS)]) for t in grid
        ]))
    return np.concatenate(stream)


def report(name: str, stream: np.ndarray, hz: float) -> dict:
    velocity = np.diff(stream, axis=0) * hz
    speed = np.linalg.norm(velocity, axis=1)
    jump = np.linalg.norm(np.diff(velocity, axis=0), axis=1)  # tick-to-tick velocity change
    row = {
        "name": name, "ticks": len(stream), "seconds": round(len(stream) / hz, 2),
        "speed_p50": round(float(np.median(speed)), 3), "speed_max": round(float(speed.max()), 3),
        "speed_ratio_p95_p05": round(float(np.percentile(speed, 95) / max(np.percentile(speed, 5), 1e-6)), 1),
        "jump_p50": round(float(np.median(jump)), 4), "jump_p99": round(float(np.percentile(jump, 99)), 3),
        "jump_max": round(float(jump.max()), 3),
    }
    print(f"  {name:<26} {row['seconds']:>7.2f}s  speed p50 {row['speed_p50']:>6.3f} max {row['speed_max']:>6.3f}"
          f"  |dv| p99 {row['jump_p99']:>7.3f} max {row['jump_max']:>8.3f} rad/s")
    return row


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg, states, actions, chunks = load(args)
    curve = ar.progress_curve(states, actions, cfg)
    frames = replan_frames(chunks, curve, cfg, args.replan_points)
    print(f"episode {args.episode}: {len(actions)} recorded frames = {len(actions) / ar.FPS:.2f} s, "
          f"{len(frames)} replans, exec {args.exec_hz:g} Hz, latency {args.latency_s * 1000:.0f} ms")

    # Limits reused from the Ruckig study, so the two demos describe the same machine.
    envelope = np.quantile(np.abs(np.diff(actions[:, :DOFS], axis=0)) * ar.FPS, 0.995, axis=0)
    envelope = np.maximum(envelope, 0.25 * float(np.median(envelope[envelope > 1e-2])))
    limits = cp.Limits(v_max=envelope, a_max=envelope / 0.15, j_max=envelope / 0.15 / 0.03)

    print("\nstrategy                     duration   commanded joint speed        tick-to-tick change")
    rows = [
        report("1 point per tick (today)", naive(chunks, frames, args.replan_points), args.exec_hz),
        report("scheduled + linear", linear(chunks, frames, args, cfg), args.exec_hz),
        report("scheduled + hermite", run_follower(chunks, frames, args), args.exec_hz),
        report("+ blend + rate limit", run_follower(chunks, frames, args, limits=limits), args.exec_hz),
    ]
    recorded_speed = np.linalg.norm(np.diff(actions[:, :DOFS], axis=0), axis=1) * ar.FPS
    print(f"  {'(the recording itself)':<26} {len(actions) / ar.FPS:>7.2f}s  "
          f"speed p50 {np.median(recorded_speed):>6.3f} max {recorded_speed.max():>6.3f}")

    figure, axes = plt.subplots(len(rows), 1, figsize=(11, 2.6 * len(rows)), sharey=True)
    streams = [naive(chunks, frames, args.replan_points), linear(chunks, frames, args, cfg),
               run_follower(chunks, frames, args), run_follower(chunks, frames, args, limits=limits)]
    for axis, row, stream, colour in zip(axes, rows, streams,
                                         ["#cf222e", "#d29922", "#1f6feb", "#2da44e"], strict=True):
        speed = np.linalg.norm(np.diff(stream, axis=0), axis=1) * args.exec_hz
        axis.plot(np.arange(len(speed)) / args.exec_hz, speed, colour, linewidth=0.8)
        axis.set_title(f"{row['name']}  --  |dv| p99 {row['jump_p99']:.2f} rad/s, "
                       f"speed spread {row['speed_ratio_p95_p05']:.0f}x", fontsize=10, loc="left")
        axis.set_ylabel("|dq/dt| (rad/s)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(f"commanded joint speed, episode {args.episode}: four ways to play the same chunks",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(args.out_dir / f"playback_ep{args.episode:03d}.png", dpi=140)
    plt.close(figure)

    (args.out_dir / f"playback_ep{args.episode:03d}.json").write_text(json.dumps(
        {"episode": args.episode, "exec_hz": args.exec_hz, "latency_s": args.latency_s,
         "replan_points": args.replan_points, "v_nominal": args.v_nominal, "dt_min": args.dt_min,
         "blend_s": args.blend_s, "strategies": rows}, indent=2))
    print(f"\nwrote {args.out_dir / f'playback_ep{args.episode:03d}.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
