"""Progress-coordinate resampling of AgiBot G2 action sequences. Pure functions, no IO.

Named for what it does rather than for how it started. `MetricConfig.metric` picks the progress
metric: "curvature" -- what this collection is built with -- spaces points by |d2q|^(1/3), so they
crowd where the trajectory bends and thin out where it runs straight. "arc" is the older,
equal-distance rule. Neither spaces points evenly in time, which is the fact anything consuming
these labels has to know: a controller that plays them at a fixed rate commands a speed that
follows the point spacing, and on this data that spans a factor of 15.

A 30 fps teleoperated episode is sampled uniformly in TIME, so the action labels a policy learns are
dense wherever the operator hesitated and sparse wherever the robot moved fast. About 19% of the
frames in this collection sit below 0.002 rad/frame and carry 0.23% of the joint-space distance while
taking 19% of the supervision. Resampling uniformly in a PROGRESS coordinate reallocates that
supervision to where the trajectory actually goes somewhere.

Progress is built as a monotone integral with two regimes:

    ds(t) = | step (x density)                  t inside a grasp event window
            | max(raw(t) - deadzone, 0)         everywhere else

where raw(t) is |d2q(t)|^(1/3) under the curvature metric, or ||dq(t)|| under the arc metric.

Away from the grasp events the samples are spaced by curvature, so a straight stretch costs few
points however far it runs; `span_frames / action_horizon` sets that density. A grasp event is
handled separately: it receives a fixed budget of points spread evenly in TIME, which preserves its
duration -- the quantity that matters there, since the vacuum takes as long as it takes.

A grasp event -- the final approach plus the vacuum dwell that follows the command -- is carried at
RECORDED TIME instead, one sample per source frame. Two separate reasons, both fatal to equal-arc
sampling there:

  * The dwell is physics. The vacuum takes a measured median of 37 to 44 frames to build, depending
    on the collection. That window carries far less joint-space distance than time, so equal-arc
    sampling compresses it -- and vacuum does not build faster because the trajectory is played
    faster. A compressed dwell teaches the policy to lift before the seal forms. The window is here
    to prevent that compression, not to add samples: at 30 fps the observations are already dense,
    and interpolating past that only invents labels between frames the policy never sees.
  * The approach has no arc to sample. The operator lines the cup up and holds still before
    triggering; in episode 0 the arm sits at 0.0001 rad/frame for the 12 frames before the command.
    A stationary segment gets no samples under any arc-based rule, at any scale, so "you are in
    position, now trigger" would never appear as a labelled state. Time is the only coordinate that
    can show it.

The step then follows in closed form. With F frames inside event windows, A the progress accumulated
outside them, and N = F + (T-1-F)/scale the output sample count:

    step = A * scale / (T - 1 - F)

`chunk_at` reads the progress of one observation frame and walks forward from there, so every frame
gets a chunk anchored on where the robot actually is rather than on a fixed grid.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.interpolate

FPS: float = 30.0

# state/action dims 0..21 are absolute joint targets in the same order. State 22/25 are the suction
# enabled flags and 23-24 / 26-27 the two vacuum channels per arm; actions 22/23 are the commands.
JOINT_DIMS: tuple[int, ...] = tuple(range(22))
SUCTION_ACTION_DIMS: tuple[int, ...] = (22, 23)
# action command dim -> the two vacuum-feedback dims of the same arm, in the state vector
VACUUM_STATE_DIMS: dict[int, tuple[int, int]] = {22: (24, 23), 23: (27, 26)}


Metric = str  # "curvature" | "arc"


@dataclasses.dataclass(frozen=True)
class MetricConfig:
    """The action space and the quantity progress is measured in.

    `metric="curvature"` spaces the samples by |q''|^(1/3), which is the density that equalises the
    error of a piecewise-cubic fit. Measured over 30 episodes, that beats equal arc length by a
    factor of three at a fixed budget -- interpolation error follows curvature, not distance, so a
    straight stretch is cheap to describe however far it runs and a corner is expensive however short
    it is. `metric="arc"` is the earlier equal-distance rule, kept so old datasets still rebuild.

    Equal weights by default: every joint counts one radian as one radian. This is deliberately the
    naive metric for a first validation. It over-counts the lower body, whose five channels
    (ankle/knee/hip pitch, hip roll/yaw) encode only two physical degrees of freedom -- this robot
    has its legs removed and sits on a lift, so those channels are a redundant parameterisation of
    "lift" and "turn". Correcting that needs two calibration numbers (lift travel in metres, radius
    from the turn centre to the end effector), not a URDF.
    """

    dims: tuple[int, ...] = JOINT_DIMS
    weights: np.ndarray | None = None  # None = equal weights
    metric: Metric = "curvature"
    # No more than one sample per source frame: the recording has nothing finer to describe, so
    # points crowded closer than this are taken from stretches that still have detail left.
    min_gap_frames: float = 1.0


@dataclasses.dataclass(frozen=True)
class SealConfig:
    """The suction dwell: from the 0->1 command to the vacuum actually holding."""

    threshold: float = 0.5  # a channel counts as sealed above this
    # Floor on the dwell, in frames. 0 uses each event's own measured seal time. A floor was tried at 45 frames, calibrated on the
    # task_2 collection (p50 44); on a later collection that seals in a p50 of 37 it bound on 83% of
    # events and stretched a typical dwell by 1.32x, which reads on video as the arm stalling. The
    # floor is a per-collection property, so hard-coding one is how it goes wrong -- the measurement
    # is already per event, and that is the number to trust.
    min_dwell_frames: int = 0
    max_dwell_frames: int = 120  # guard against a run-on window if a channel never crosses


@dataclasses.dataclass(frozen=True)
class ProgressConfig:
    # Off by default. Equal-arc sampling already starves a pause of samples all by itself, because a
    # pause contributes almost no arc: measured, turning the dead zone off moves the supervision that
    # lands on pause frames from 0.52% to 0.72% (picking) and 0.71% to 1.00% (placement), and changes
    # neither the speedup nor the event share. It was solving a problem the metric had already
    # solved, so it is one fewer threshold to calibrate per collection.
    deadzone: float = 0.0
    # The final approach, carried at recorded time like the dwell that follows it. Equal-arc sampling
    # cannot serve this window: the operator lines the cup up and holds still before triggering --
    # in episode 0 the arm is frozen at 0.0001 rad/frame for the 12 frames before the command -- and
    # a segment with no arc length gets no samples no matter how much it is scaled. Time is the only
    # coordinate under which "you are in position, now trigger" is a state the policy can be shown.
    approach_frames: int = 15
    # Samples given to one grasp event -- the approach plus the vacuum transient, taken together --
    # capped at one per frame, since the recording holds nothing finer.
    # Earlier this was one sample per source frame, which measured out at roughly six times more than
    # the window needs: reconstructing an event from 8 points already lands at 2.0 mm, under the
    # 1.97 mm jitter floor, while one-per-frame spends about 52. The points are spread uniformly in
    # TIME across the window, which is what preserves its duration -- a curvature rule would give the
    # dwell almost nothing, since the arm barely moves while the vacuum builds.
    event_points: int = 32
    seal: SealConfig = dataclasses.field(default_factory=SealConfig)


@dataclasses.dataclass(frozen=True)
class ResampleConfig:
    metric: MetricConfig = dataclasses.field(default_factory=MetricConfig)
    progress: ProgressConfig = dataclasses.field(default_factory=ProgressConfig)
    # How far ahead one chunk of `action_horizon` points should reach, in recorded frames. This is
    # the quantity the design is actually about; the density that produces it -- span/horizon frames
    # per step in the free stretches -- is derived rather than configured. An earlier version exposed
    # that density directly, under a name left over from when it meant execution speed, which stated
    # an implementation detail instead of the intent.
    #
    # Coverage still floats below `span_frames` near an event, because the event window spends its points
    # over a short stretch. A chunk that ends inside an event is not a problem: the next observation
    # continues from there.
    span_frames: int = 200
    action_horizon: int = 50

    @property
    def frames_per_step(self) -> float:
        """Free-stretch density implied by the span and the horizon."""
        return self.span_frames / self.action_horizon


@dataclasses.dataclass(frozen=True)
class DwellWindow:
    """One suction command edge and the vacuum transient that follows it.

    Both edges matter and neither is instantaneous. A grasp (0->1) is not done until the vacuum has
    built; a release (1->0) is not done until it has bled away, and until then the part is still
    attached. A collection may hold only one kind -- the picking set has 150 grasps and no releases,
    the placement set 136 releases and no grasps -- so looking for only one leaves the other
    collection with no protected window at all.
    """

    action_dim: int
    kind: str  # "grasp" | "release"
    onset: int
    sealed: int  # first frame at or after onset where a channel crosses the threshold
    end: int  # onset + dwell, where dwell = max(sealed - onset, min_dwell_frames)

    @property
    def measured_frames(self) -> int:
        return self.sealed - self.onset

    @property
    def frames(self) -> int:
        return self.end - self.onset


@dataclasses.dataclass(frozen=True)
class ProgressCurve:
    """s(t) on the original frame grid, plus the pieces it was built from."""

    s: np.ndarray  # (T,) monotone increasing, s[0] = 0
    ds: np.ndarray  # (T-1,)
    raw_step: np.ndarray  # (T-1,) ||dq|| before the dead zone and the event windows
    dwell_mask: np.ndarray  # (T-1,) True inside a suction dwell
    approach_mask: np.ndarray  # (T-1,) True inside the approach that precedes one
    step: float  # the uniform spacing the grid uses
    windows: tuple[DwellWindow, ...]

    @property
    def total(self) -> float:
        return float(self.s[-1])

    @property
    def n_samples(self) -> int:
        """Output sample count, so that an episode of T frames yields T-1 -> n_samples steps."""
        return int(round(self.total / self.step))


@dataclasses.dataclass(frozen=True)
class Resampled:
    t: np.ndarray  # (N,) fractional source-frame index of each sample
    s: np.ndarray  # (N,) its progress coordinate
    actions: np.ndarray  # (N, D)


def config_from_report(report: dict) -> ResampleConfig:
    """Rebuild the config a dataset was built with, from the `meta/arc_resample.json` it carries.

    Analysis scripts must not fall back on the module defaults: those change as the method is tuned,
    and a dataset on disk was built with whatever was current that day. Reading the provenance file
    is what keeps a re-analysis about the data rather than about today's constants.
    """
    cfg = report["config"]
    progress = ProgressConfig(
        deadzone=cfg["deadzone"],
        approach_frames=cfg["approach_frames"],
        seal=SealConfig(min_dwell_frames=cfg["min_dwell_frames"]),
    )
    if "event_points" in cfg:
        progress = dataclasses.replace(progress, event_points=cfg["event_points"])
    span = cfg.get("span_frames")
    if span is None:  # datasets built before the span/horizon reparameterisation
        span = cfg["speed_scale"] * cfg["action_horizon"]
    return ResampleConfig(
        progress=progress,
        metric=MetricConfig(metric=cfg.get("metric", "arc")),
        span_frames=int(round(span)),
        action_horizon=cfg["action_horizon"],
    )


def frozen_prefix(states: np.ndarray, actions: np.ndarray, eps: float = 1e-3) -> int:
    """Leading frames where the arm has neither moved nor been told to move.

    Every recording here opens with the camera already rolling and the operator not yet started --
    three seconds of a parked arm, a fifth of the episode. Those rows carry no information: the joints
    are constant and the three camera streams differ only by sensor and codec noise, so they are the
    same training sample repeated a hundred times.

    Both traces have to be quiet. Trimming on the measured joints alone would run past the operator's
    first command on any lag between commanding and moving, and that command is the one frame of the
    prefix that matters.
    """
    measured = np.linalg.norm(np.diff(states[:, JOINT_DIMS], axis=0), axis=1)
    commanded = np.linalg.norm(np.diff(actions[:, JOINT_DIMS], axis=0), axis=1)
    moving = np.flatnonzero(np.maximum(measured, commanded) > eps)
    return int(moving[0]) if len(moving) else len(states)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [lo, hi) spans of consecutive True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(edges > 0).tolist(), np.flatnonzero(edges < 0).tolist(), strict=True))


def _transitions(channel: np.ndarray) -> np.ndarray:
    """Frame indices where a binary channel goes 0->1 (the index of the first ON frame)."""
    return np.flatnonzero(np.diff((channel > 0.5).astype(np.int8)) > 0) + 1


def dwell_windows(states: np.ndarray, actions: np.ndarray, cfg: SealConfig) -> tuple[DwellWindow, ...]:
    """Locate every suction transient, measuring each from its own vacuum trace.

    Grasp windows run from the command going on until the vacuum first holds; release windows from
    the command going off until it has bled below the same threshold.
    """
    n = len(states)
    out: list[DwellWindow] = []
    for cmd_dim, (ch1, ch2) in VACUUM_STATE_DIMS.items():
        vacuum = np.maximum(states[:, ch1], states[:, ch2])
        edges = [("grasp", _transitions(actions[:, cmd_dim]), True),
                 ("release", _transitions(1.0 - actions[:, cmd_dim]), False)]
        for kind, onsets, want_above in edges:
          for onset in onsets:
            reached = vacuum[onset:] > cfg.threshold if want_above else vacuum[onset:] <= cfg.threshold
            crossed = np.flatnonzero(reached)
            measured = int(crossed[0]) if len(crossed) else cfg.max_dwell_frames
            dwell = min(max(measured, cfg.min_dwell_frames), cfg.max_dwell_frames)
            out.append(
                DwellWindow(
                    action_dim=cmd_dim,
                    kind=kind,
                    onset=int(onset),
                    sealed=int(onset) + measured,
                    end=min(int(onset) + dwell, n - 1),
                )
            )
    return tuple(sorted(out, key=lambda w: w.onset))


def progress_curve(states: np.ndarray, actions: np.ndarray, cfg: ResampleConfig) -> ProgressCurve:
    """Build s(t) and the uniform step the resampling grid walks in."""
    q = actions[:, cfg.metric.dims].astype(np.float64)
    if cfg.metric.weights is not None:
        q = q * cfg.metric.weights
    if cfg.metric.metric == "curvature":
        second = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
        raw = np.power(np.concatenate([second[:1], second]) + 1e-12, 1.0 / 3.0)
    else:
        raw = np.linalg.norm(np.diff(q, axis=0), axis=1)
    n = len(raw)

    windows = dwell_windows(states, actions, cfg.progress.seal)
    approach = np.zeros(n, dtype=bool)
    dwell = np.zeros(n, dtype=bool)
    for w in windows:
        approach[max(w.onset - cfg.progress.approach_frames, 0) : w.onset] = True
    for w in windows:
        # Applied after every approach so that a dwell overlapping a later approach wins: the dwell
        # is a physical duration and must not be reshaped by a neighbouring window.
        dwell[w.onset : min(w.end, n)] = True
    approach &= ~dwell
    event = approach | dwell

    free = np.maximum(raw - cfg.progress.deadzone, 0.0)
    free[event] = 0.0

    n_free = n - int(event.sum())
    total_free = float(free.sum())
    if n_free <= 0 or total_free <= 0.0:
        # Degenerate episode (all event, or no motion at all): fall back to equal time.
        step = max(total_free, 1.0) / max(n, 1)
    else:
        step = total_free * cfg.frames_per_step / n_free

    # Each contiguous run of event frames gets the same budget, spread evenly in time across it, so
    # its duration survives however long or short the run happens to be -- but never more points than
    # the run has frames. Half of these windows are 16 frames long (a placement, whose vacuum bleeds
    # away in one frame, plus the 15-frame approach), and a flat budget of 32 would put two points on
    # every frame: duplicate labels, describing nothing the recording does not already say once.
    ds = free.copy()
    for lo, hi in _runs(event):
        frames = max(hi - lo, 1)
        ds[lo:hi] = step * min(cfg.progress.event_points, frames) / frames
    return ProgressCurve(
        s=np.concatenate([[0.0], np.cumsum(ds)]),
        ds=ds,
        raw_step=raw,
        dwell_mask=dwell,
        approach_mask=approach,
        step=float(step),
        windows=windows,
    )


def invert(curve: ProgressCurve, targets: np.ndarray) -> np.ndarray:
    """Map progress values back to fractional frame indices.

    s has plateaus wherever the dead zone zeroed a pause, so it is not invertible there. A negligible
    drift makes it strictly increasing; a target landing inside a plateau then resolves to somewhere
    inside the pause, and since the plateau spans essentially no progress at all, the grid steps over
    it -- which is the point.
    """
    t = np.arange(len(curve.s), dtype=np.float64)
    strict = curve.s + np.arange(len(curve.s)) * (max(curve.total, 1.0) * 1e-12)
    return np.interp(targets, strict, t)


def sample_actions(actions: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the action vector at fractional frame indices.

    Continuous dims go through PCHIP: it passes through every recorded sample and, being
    shape-preserving, cannot overshoot into oscillations the arm never made. The suction command is
    held at the nearest frame and re-binarised -- the valve takes 0 or 1.
    """
    continuous = [d for d in range(actions.shape[1]) if d not in SUCTION_ACTION_DIMS]
    grid = np.arange(len(actions), dtype=np.float64)
    curve = scipy.interpolate.PchipInterpolator(grid, actions[:, continuous], axis=0)

    out = np.zeros((len(t), actions.shape[1]), dtype=np.float64)
    out[:, continuous] = curve(np.clip(t, 0.0, len(actions) - 1.0))
    nearest = np.clip(np.round(t).astype(int), 0, len(actions) - 1)
    out[:, SUCTION_ACTION_DIMS] = np.round(actions[nearest][:, SUCTION_ACTION_DIMS])
    return out


def resample(states: np.ndarray, actions: np.ndarray, cfg: ResampleConfig) -> tuple[Resampled, ProgressCurve]:
    """Resample a whole episode uniformly in progress. Used for reporting, not for the dataset."""
    curve = progress_curve(states, actions, cfg)
    targets = np.arange(curve.n_samples + 1, dtype=np.float64) * curve.step
    t = invert(curve, np.clip(targets, 0.0, curve.total))
    return Resampled(t=t, s=targets, actions=sample_actions(actions, t)), curve


def chunk_at(frame: int, actions: np.ndarray, curve: ProgressCurve, horizon: int) -> np.ndarray:
    """The (horizon, D) action chunk for the observation at `frame`.

    The grid starts at this observation's own progress, not at a multiple of the step, so the chunk
    is anchored on where the robot actually is. Past the end of the episode the samples clamp to the
    final frame, which is how a fixed-length chunk is padded today.
    """
    targets = curve.s[frame] + np.arange(1, horizon + 1, dtype=np.float64) * curve.step
    t = invert(curve, np.clip(targets, 0.0, curve.total))
    return sample_actions(actions, t)


def all_chunks(actions: np.ndarray, curve: ProgressCurve, horizon: int) -> np.ndarray:
    """Every frame's chunk, as (T, horizon, D). One PCHIP fit for the whole episode."""
    n = len(actions)
    targets = curve.s[:, None] + np.arange(1, horizon + 1, dtype=np.float64)[None, :] * curve.step
    t = invert(curve, np.clip(targets.ravel(), 0.0, curve.total))
    return sample_actions(actions, t).reshape(n, horizon, actions.shape[1])


def all_chunk_weights(curve: ProgressCurve, horizon: int, key_weight: float) -> np.ndarray:
    """Per-chunk-step loss multipliers, as (T, horizon).

    A step is upweighted when the source frame it was drawn from lies inside an event window. The
    windows already run at recorded time, but that only stops them being COMPRESSED -- it no longer
    makes them denser than their surroundings, so on its own it gives the grasp no extra pull on the
    loss. Roughly 10% of frames carry the moment the task succeeds or fails; this is what says so.
    """
    n = len(curve.s)
    targets = curve.s[:, None] + np.arange(1, horizon + 1, dtype=np.float64)[None, :] * curve.step
    t = invert(curve, np.clip(targets.ravel(), 0.0, curve.total))
    index = np.clip(np.rint(t).astype(int), 0, len(curve.ds) - 1)
    inside = (curve.approach_mask | curve.dwell_mask)[index].reshape(n, horizon)
    return np.where(inside, key_weight, 1.0)
