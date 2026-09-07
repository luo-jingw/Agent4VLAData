"""How few keypoints reconstruct the trajectory, and where they should go.

The action chunk currently carries 50 samples spanning about 1.9 s. That is a choice about density,
not a measurement: nobody has asked how many points the trajectory actually needs. If a whole
16-second episode can be rebuilt from 60 keypoints, then the same 50 slots could describe the entire
task instead of one and a half seconds of it, and the policy would be predicting a plan rather than a
segment.

The question is rate-distortion. Given N points and PCHIP between them -- the same interpolator the
labels already use -- how far does the reconstruction sit from the recording? And since placement
matters as much as count, four ways of choosing the points are compared:

  time      every T/N frames. What the raw recording gives.
  arc       equal joint-space distance. What the current labels do.
  curvature density proportional to |q''|^(1/3), the spacing that equalises interpolation error for a
            piecewise-cubic fit rather than equalising distance.
  greedy    start from the endpoints, repeatedly insert wherever the error is currently worst. Not a
            practical rule -- it needs the whole episode in advance -- but it bounds what any
            placement rule could achieve, which is what makes the other three readable.

Error is reported in joint space and, through the URDF, at the suction cup, because millimetres are
what the task cares about.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python keypoint_budget.py --episodes 6
"""
from __future__ import annotations
import dataclasses, json, pathlib
import progress_resample as ar, eef_error as fk, numpy as np, pandas as pd, scipy.interpolate, tyro

ROOT = pathlib.Path(__file__).resolve().parent
J = list(ar.JOINT_DIMS)

@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v1"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "keypoint_budget.json"
    episodes: int = 6
    counts: tuple[int, ...] = (10, 15, 20, 30, 40, 60, 80, 120, 200)
    recon: str = "linear"          # how the controller joins the emitted points
    curv_exponent: float = 0.5     # 1/2 for a linear controller, 1/3 for a cubic one

RECON = "pchip"  # module-level so every helper follows the same rule; set by main()


def reconstruct(q: np.ndarray, idx: np.ndarray, how: str | None = None) -> np.ndarray:
    """Rebuild the full trajectory from the rows at `idx`.

    `linear` is what a controller does by default: it receives the emitted points and drives straight
    between them. `pchip` is what the labels are built with, and is only what execution looks like if
    the inference side is changed to interpolate the same way. Scoring with pchip while the robot
    walks straight lines flatters the representation, so linear is the honest default.
    """
    how = how or RECON
    idx = np.unique(np.clip(np.rint(idx).astype(int), 0, len(q) - 1))
    if len(idx) < 2:
        idx = np.array([0, len(q) - 1])
    grid = np.arange(len(q), dtype=float)
    if how == "linear":
        return np.stack([np.interp(grid, idx.astype(float), q[idx, j]) for j in range(q.shape[1])], axis=1)
    return scipy.interpolate.PchipInterpolator(idx.astype(float), q[idx], axis=0)(grid)

def by_time(q: np.ndarray, n: int) -> np.ndarray:
    return np.linspace(0, len(q) - 1, n)

def by_arc(q: np.ndarray, n: int) -> np.ndarray:
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    s = s + np.arange(len(s)) * 1e-12
    return np.interp(np.linspace(0, s[-1], n), s, np.arange(len(s), dtype=float))

def by_curvature(q: np.ndarray, n: int) -> np.ndarray:
    """Density proportional to |q''|^(1/3), capped at one keypoint per source frame."""
    return by_curvature_of(q, n)

def greedy_order_eef(q: np.ndarray, truth_cup: np.ndarray, chain, up_to: int) -> list[int]:
    """Greedy insertion scored at the cup instead of in joint space."""
    idx = [0, len(q) - 1]
    order = list(idx)
    while len(idx) < up_to:
        err = np.linalg.norm(batched_cup(chain, reconstruct(q, np.array(idx))) - truth_cup, axis=1)
        err[idx] = -1.0
        pick = int(np.argmax(err))
        idx = sorted([*idx, pick])
        order.append(pick)
    return order


def greedy_order(q: np.ndarray, up_to: int) -> list[int]:
    """Insertion order for the greedy rule, built once. A prefix of it is the answer for every N."""
    idx = [0, len(q) - 1]
    order = list(idx)
    while len(idx) < up_to:
        err = np.linalg.norm(reconstruct(q, np.array(idx)) - q, axis=1)
        err[idx] = -1.0
        pick = int(np.argmax(err))
        idx = sorted([*idx, pick])
        order.append(pick)
    return order

def _lsq(q: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Least-squares cubic B-spline with the given interior knots, evaluated at every frame.

    This is fitting rather than interpolation: the control points are free to sit off the trajectory.
    Interpolation through samples is the special case where they are pinned to it, so for the same
    budget this can only do better -- and for a fixed basis the optimum is a linear least-squares
    solve, not a search.
    """
    t = np.arange(len(q), dtype=float)
    k = 3
    knots = np.concatenate([np.full(k + 1, t[0]), interior, np.full(k + 1, t[-1])])
    spline = scipy.interpolate.make_lsq_spline(t, q, knots, k=k)
    return spline(t)


def _interior_from(positions: np.ndarray, n: int, last: float, min_gap: float = 1.5) -> np.ndarray:
    """Trim a candidate set down to the interior knots a cubic with `n` coefficients needs.

    A least-squares spline needs data inside every basis function's support (the Schoenberg-Whitney
    condition), so knots may not crowd closer than the samples themselves. The curvature rule happily
    stacks them at a sharp corner, which makes the normal equations singular and the fit come back as
    NaN -- hence the minimum gap, and hence a fitted budget that can be smaller than asked for.
    """
    want = max(n - 4, 0)
    if want == 0:
        return np.empty(0)
    inner = np.unique(np.clip(positions, 1e-6, last - 1e-6))
    if len(inner) > want:
        inner = inner[np.linspace(0, len(inner) - 1, want).round().astype(int)]
    kept = []
    for x in inner:
        if not kept or x - kept[-1] >= min_gap:
            kept.append(float(x))
    return np.array(kept)


def by_fit_uniform(q: np.ndarray, n: int) -> np.ndarray:
    """Marker for the fitted strategies; the reconstruction is produced in `evaluate`."""
    return by_time(q, n)


def by_arc_of(path: np.ndarray, n: int) -> np.ndarray:
    """Equal arc length along whatever path is handed in -- joint space or the cup's."""
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)]) + np.arange(len(path)) * 1e-12
    return np.interp(np.linspace(0, s[-1], n), s, np.arange(len(path), dtype=float))


CURV_EXP = 1.0 / 3.0  # set by main(); 1/3 equalises cubic error, 1/2 equalises linear error


def by_curvature_of(path: np.ndarray, n: int, min_gap: float = 1.0, exponent: float | None = None) -> np.ndarray:
    """Density proportional to |p''|^(1/3), capped at one keypoint per source frame.

    Without the cap the rule will happily stack several keypoints inside a single frame interval
    wherever curvature spikes. Those extra points carry no information -- the recording only has one
    sample there -- and they are taken from the budget at the expense of stretches that do have
    detail left to describe. `min_gap` is that cap, in source frames; the points it frees are pushed
    outward rather than dropped, so the requested count is still met.
    """
    second = np.linalg.norm(np.diff(path, n=2, axis=0), axis=1)
    density = np.power(np.concatenate([[second[0]], second, [second[-1]]]) + 1e-12,
                       CURV_EXP if exponent is None else exponent)
    c = np.concatenate([[0.0], np.cumsum(density)]) + np.arange(len(path) + 1) * 1e-12
    idx = np.interp(np.linspace(0, c[-1], n), c, np.arange(len(c), dtype=float))
    if min_gap <= 0:
        return idx
    # One forward pass, then one backward pass, so the cap does not bunch everything at the tail.
    last = float(len(path) - 1)
    out = idx.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + min_gap)
    excess = out[-1] - last
    if excess > 0:
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - min_gap)
        out = np.clip(out, 0.0, last)
    return out


def curvature_crowding(path: np.ndarray, n: int) -> float:
    """Share of the requested keypoints the uncapped rule would place closer than one frame apart."""
    idx = by_curvature_of(path, n, min_gap=0.0)
    return float((np.diff(idx) < 1.0).mean())


STRATEGIES = {"time": by_time, "arc": by_arc, "curvature": by_curvature}
# Placement decided in the space the error is finally measured in. The stored values stay joint
# angles either way -- the policy has to emit joint commands -- so only the choice of WHERE to put a
# keypoint moves into Cartesian space.
EEF_RULES = ("arc-eef", "curv-eef", "greedy-eef")
FITTED = ("fit-uniform", "fit-curvature")


def evaluate(name, q: np.ndarray, n: int, order: list[int] | None = None) -> np.ndarray:
    """Reconstruction for one strategy at one budget."""
    last = float(len(q) - 1)
    if name == "greedy":
        return reconstruct(q, np.array(sorted(order[:n]), dtype=float))
    if name in ("fit-uniform", "fit-curvature"):
        raw = np.linspace(0, last, n)[1:-1] if name == "fit-uniform" else by_curvature(q, n)[1:-1]
        knots = _interior_from(raw, n, last)
        try:
            return _lsq(q, knots)
        except (ValueError, np.linalg.LinAlgError):
            # Thin further and retry once; report by falling back rather than emitting NaN.
            return _lsq(q, _interior_from(raw, n, last, min_gap=3.0))
    return reconstruct(q, STRATEGIES[name](q, n))

def batched_cup(chain, poses: np.ndarray) -> np.ndarray:
    """Cup position for every row of `poses`, walking the chain once over the whole batch.

    The per-pose version calls forward() T times and dominates the runtime as soon as a criterion has
    to be scored in Cartesian space. Here each of the chain's ~16 transforms is applied to all T poses
    at once, which is what makes an end-effector-space greedy search affordable at all.
    """
    n = len(poses)
    pos = np.zeros((n, 3))
    rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    col = {fk.DIM_TO_JOINT[d]: i for i, d in enumerate(J)}
    for link in chain:
        pos = pos + rot @ link.translation
        rot = rot @ link.rotation
        if link.axis is None:
            continue
        angles = poses[:, col[link.name]] if link.name in col else np.zeros(n)
        k = link.axis / np.linalg.norm(link.axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        c, sn = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
        step = np.eye(3) + sn * K + (1 - c) * (K @ K)
        rot = rot @ step
    return pos


def main(args: Args) -> None:
    global RECON, CURV_EXP
    RECON, CURV_EXP = args.recon, args.curv_exponent
    print(f"reconstruction: {args.recon}   curvature exponent: {args.curv_exponent}")
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    def cup(poses: np.ndarray) -> np.ndarray:
        return batched_cup(chain, poses)

    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]
    names = [*STRATEGIES, "greedy", *FITTED, *EEF_RULES]
    acc: dict[str, dict[int, list]] = {k: {n: [] for n in args.counts} for k in names}
    frames = []

    for path in picks:
        q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
        truth_cup = cup(q)
        frames.append(len(q))
        order = greedy_order(q, max(args.counts))
        order_eef = greedy_order_eef(q, truth_cup, chain, max(args.counts))
        for name in names:
            for n in args.counts:
                if name == "arc-eef":
                    rec = reconstruct(q, by_arc_of(truth_cup, n))
                elif name == "curv-eef":
                    rec = reconstruct(q, by_curvature_of(truth_cup, n))
                elif name == "greedy-eef":
                    rec = reconstruct(q, np.array(sorted(order_eef[:n]), dtype=float))
                else:
                    rec = evaluate(name, q, n, order)
                joint = np.linalg.norm(rec - q, axis=1)
                mm = np.linalg.norm(cup(rec) - truth_cup, axis=1) * 1000
                acc[name][n].append((float(np.percentile(joint, 95)), float(joint.max()),
                                     float(np.percentile(mm, 95)), float(mm.max())))

    summary = {"episodes": len(picks), "frames_median": int(np.median(frames)), "counts": list(args.counts),
               "strategies": {}}
    for name in names:
        rows = {}
        for n in args.counts:
            a = np.array(acc[name][n])
            rows[n] = {"joint_p95_mrad": round(float(np.median(a[:, 0])) * 1000, 3),
                       "cup_p95_mm_median": round(float(np.median(a[:, 2])), 3),
                       "cup_p95_mm_p10": round(float(np.percentile(a[:, 2], 10)), 3),
                       "cup_p95_mm_p90": round(float(np.percentile(a[:, 2], 90)), 3),
                       "cup_max_mm_median": round(float(np.median(a[:, 3])), 3),
                       "episodes": len(a)}
        summary["strategies"][name] = rows

    print(f"{len(picks)} episodes, median {int(np.median(frames))} frames each")
    print(f"\n{'N':>5} " + " ".join(f"{k:>20}" for k in names))
    for n in args.counts:
        cells = [f"{summary['strategies'][k][n]['cup_p95_mm_median']:6.1f} "
                 f"[{summary['strategies'][k][n]['cup_p95_mm_p10']:5.1f}-{summary['strategies'][k][n]['cup_p95_mm_p90']:5.1f}]"
                 for k in names]
        print(f"{n:5d} " + " ".join(f"{c:>20}" for c in cells))
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main(tyro.cli(Args))
