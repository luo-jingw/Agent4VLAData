"""What the controller actually does between keypoints, and what that changes.

Every reconstruction so far used PCHIP, because that is what builds the labels. Execution is not
obliged to agree: hand a robot 50 waypoints and, unless its controller is changed, it walks straight
lines between them. Measuring with PCHIP therefore measures a trajectory nobody executes.

Two things follow, and only the first is obvious:

  * Straight lines cut corners the cubic would round, so the error grows. How much is worth knowing,
    because it prices the change to the controller.
  * The optimal PLACEMENT changes with the interpolator. Interpolation error goes as h^(k+1) for
    degree k, so equalising it across intervals wants density proportional to |q^(k+1)|^(1/(k+1)):
    a different exponent, and a different derivative, for lines than for cubics. The exponent in use
    (1/3 on the second derivative) was chosen for the cubic. Rather than trust the theory, several
    exponents are measured under both interpolators.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python linear_interp_budget.py --episodes 30
"""
from __future__ import annotations
import dataclasses, json, pathlib
import eef_error as fk, keypoint_budget as kb, numpy as np, pandas as pd, scipy.interpolate, tyro
import progress_resample as ar

ROOT = pathlib.Path(__file__).resolve().parent
J = list(ar.JOINT_DIMS)

@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v2"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "linear_interp_budget.json"
    episodes: int = 30
    counts: tuple[int, ...] = (20, 30, 40, 60, 80, 120)
    exponents: tuple[float, ...] = (0.5, 1 / 3, 0.25)

def rebuild(q: np.ndarray, idx: np.ndarray, kind: str) -> np.ndarray:
    idx = np.unique(np.clip(np.rint(idx).astype(int), 0, len(q) - 1))
    if len(idx) < 2:
        idx = np.array([0, len(q) - 1])
    t = np.arange(len(q), dtype=float)
    if kind == "linear":
        return np.stack([np.interp(t, idx.astype(float), q[idx, j]) for j in range(q.shape[1])], axis=1)
    return scipy.interpolate.PchipInterpolator(idx.astype(float), q[idx], axis=0)(t)

def by_exponent(q: np.ndarray, n: int, power: float) -> np.ndarray:
    """Density proportional to |q''|^power, capped at one keypoint per source frame."""
    second = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
    density = np.power(np.concatenate([second[:1], second, second[-1:]]) + 1e-12, power)
    c = np.concatenate([[0.0], np.cumsum(density)]) + np.arange(len(q) + 1) * 1e-12
    idx = np.interp(np.linspace(0, c[-1], n), c, np.arange(len(c), dtype=float))
    last = float(len(q) - 1)
    for i in range(1, len(idx)):
        idx[i] = max(idx[i], idx[i - 1] + 1.0)
    if idx[-1] > last:
        for i in range(len(idx) - 2, -1, -1):
            idx[i] = min(idx[i], idx[i + 1] - 1.0)
    return np.clip(idx, 0.0, last)

def greedy_for(q: np.ndarray, truth: np.ndarray, chain, up_to: int, kind: str) -> list[int]:
    idx = [0, len(q) - 1]; order = list(idx)
    while len(idx) < up_to:
        err = np.linalg.norm(kb.batched_cup(chain, rebuild(q, np.array(idx), kind)) - truth, axis=1)
        err[idx] = -1.0
        pick = int(np.argmax(err)); idx = sorted([*idx, pick]); order.append(pick)
    return order

def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]
    rules = ["time", *[f"curv^{e:.2f}" for e in args.exponents], "greedy"]
    acc = {k: {r: {n: [] for n in args.counts} for r in rules} for k in ("linear", "pchip")}

    for path in picks:
        q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
        truth = kb.batched_cup(chain, q)
        for kind in ("linear", "pchip"):
            order = greedy_for(q, truth, chain, max(args.counts), kind)
            for rule in rules:
                for n in args.counts:
                    if rule == "time":
                        idx = np.linspace(0, len(q) - 1, n)
                    elif rule == "greedy":
                        idx = np.array(sorted(order[:n]), dtype=float)
                    else:
                        idx = by_exponent(q, n, float(rule.split("^")[1]))
                    rec = kb.batched_cup(chain, rebuild(q, idx, kind))
                    acc[kind][rule][n].append(float(np.percentile(np.linalg.norm(rec - truth, axis=1) * 1000, 95)))

    summary = {"episodes": len(picks),
               "results": {k: {r: {n: round(float(np.median(v)), 2) for n, v in rows.items()}
                               for r, rows in acc[k].items()} for k in acc}}
    print(f"{len(picks)} episodes. cup reconstruction error, p95 per episode, median (mm)\n")
    for kind in ("linear", "pchip"):
        print(f"{kind.upper()} interpolation between keypoints")
        print(f"  {'N':>5} " + " ".join(f"{r:>12}" for r in rules))
        for n in args.counts:
            print(f"  {n:5d} " + " ".join(f"{summary['results'][kind][r][n]:12.2f}" for r in rules))
        print()
    print("cost of straight lines (linear / pchip, best rule at each N):")
    for n in args.counts:
        bl = min(summary["results"]["linear"][r][n] for r in rules if r != "greedy")
        bp = min(summary["results"]["pchip"][r][n] for r in rules if r != "greedy")
        print(f"  N={n:4d}  linear {bl:7.2f}   pchip {bp:7.2f}   x{bl / bp:.2f}")
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main(tyro.cli(Args))
