"""Represent the trajectory in joint space or in the cup's own space -- which is cheaper?

Everything so far stored joint angles at the keypoints and interpolated between them in joint space,
using the cup only to score the result. The other option is to store the cup's position and
interpolate the Cartesian path directly. Same strategies, same error metric, different space.

Two views, because they answer different questions:

  same N            how many keypoints does each representation need? A keypoint costs 22 numbers in
                    joint space and 3 in Cartesian, so this view flatters Cartesian.
  same payload      how many numbers does each representation need? This is the one that matters for
                    a policy whose output width is fixed: the current chunk is 50 x 24 = 1200 floats,
                    which buys 50 joint keypoints or 400 Cartesian ones.

What this does NOT settle: a Cartesian path is not an action. The robot takes joint commands, so a
cup-space representation needs inverse kinematics, and 12 joints driving a 3-dof position leaves a
large null space -- the same cup path can be walked with very different arm configurations, and
nothing in a position-only representation says which. Orientation is missing too, and a suction cup
has to meet the surface square. Treat the Cartesian column as an upper bound on what a Cartesian
representation could achieve, not as a drop-in alternative.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python eef_space_budget.py --episodes 30
"""
from __future__ import annotations
import dataclasses, json, pathlib
import progress_resample as ar, eef_error as fk, keypoint_budget as kb, numpy as np, pandas as pd, tyro

ROOT = pathlib.Path(__file__).resolve().parent
J = list(ar.JOINT_DIMS)

@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v1"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "eef_space_budget.json"
    episodes: int = 30
    counts: tuple[int, ...] = (10, 20, 30, 40, 60, 80, 120, 200)

def place(name: str, path: np.ndarray, n: int, order: list[int]) -> np.ndarray:
    if name == "time": return kb.by_time(path, n)
    if name == "arc": return kb.by_arc_of(path, n)
    if name == "curvature": return kb.by_curvature_of(path, n)
    if name == "greedy": return np.array(sorted(order[:n]), dtype=float)
    raise ValueError(name)

RULES = ("time", "arc", "curvature", "greedy")

def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]

    acc = {sp: {r: {n: [] for n in args.counts} for r in RULES} for sp in ("joint", "eef")}
    for path in picks:
        q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
        cup = kb.batched_cup(chain, q)
        # Greedy is space-specific: it inserts where THAT space's own reconstruction is worst.
        order_j = kb.greedy_order_eef(q, cup, chain, max(args.counts))
        order_e = greedy_on(cup, max(args.counts))
        for rule in RULES:
            for n in args.counts:
                idx_j = place(rule, q if rule != "greedy" else cup, n, order_j)
                err_j = np.linalg.norm(kb.batched_cup(chain, kb.reconstruct(q, idx_j)) - cup, axis=1) * 1000
                idx_e = place(rule, cup, n, order_e)
                err_e = np.linalg.norm(kb.reconstruct(cup, idx_e) - cup, axis=1) * 1000
                acc["joint"][rule][n].append(float(np.percentile(err_j, 95)))
                acc["eef"][rule][n].append(float(np.percentile(err_e, 95)))

    summary = {"episodes": len(picks), "counts": list(args.counts),
               "floats_per_keypoint": {"joint": 22, "eef": 3},
               "results": {sp: {r: {n: round(float(np.median(v)), 3) for n, v in rows.items()}
                                for r, rows in acc[sp].items()} for sp in acc}}
    print(f"{len(picks)} episodes.  cup-position reconstruction error, p95 per episode, median over episodes (mm)\n")
    print("SAME NUMBER OF KEYPOINTS")
    print(f"{'N':>5} " + " ".join(f"{r+' J':>12}{r+' E':>12}" for r in RULES))
    for n in args.counts:
        print(f"{n:5d} " + " ".join(f"{summary['results']['joint'][r][n]:12.1f}{summary['results']['eef'][r][n]:12.1f}"
                                   for r in RULES))
    print("\nSAME PAYLOAD (floats).  joint keypoints x22, cup keypoints x3")
    print(f"{'floats':>7} {'N joint':>8} {'N cup':>7} " + " ".join(f"{r:>10}" for r in RULES) + "   <- cup-space at that N")
    for n in args.counts:
        floats = n * 22
        n_cup = floats // 3
        cells = []
        for r in RULES:
            xs = sorted(args.counts)
            ys = [summary["results"]["eef"][r][x] for x in xs]
            cells.append(np.interp(min(n_cup, xs[-1]), xs, ys))
        print(f"{floats:7d} {n:8d} {n_cup:7d} " + " ".join(f"{c:10.2f}" for c in cells))
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

def greedy_on(path: np.ndarray, up_to: int) -> list[int]:
    idx = [0, len(path) - 1]; order = list(idx)
    while len(idx) < up_to:
        err = np.linalg.norm(kb.reconstruct(path, np.array(idx)) - path, axis=1)
        err[idx] = -1.0
        pick = int(np.argmax(err)); idx = sorted([*idx, pick]); order.append(pick)
    return order

if __name__ == "__main__":
    main(tyro.cli(Args))
