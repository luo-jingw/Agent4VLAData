"""Chunks are re-planned every frame, so a placement rule has to be judged on consistency too.

The keypoint study measured one placement over a whole episode. Training and deployment do something
different: every frame the policy emits H keypoints covering the road ahead, and the controller
follows them until the next chunk replaces them. Two ways to place the keypoints inside such a
window, and they fail differently:

  per-window  place them by curvature within this window alone. Each chunk is as good as it can be,
              but consecutive chunks discretise the same stretch of trajectory differently, so the
              plan the controller is following changes shape on every replan.
  global      place them once along the episode and let each chunk take the next H. Consecutive
              chunks agree wherever they overlap, at the cost of the first keypoint sitting an
              arbitrary distance ahead of the observation instead of a fixed one.

Both are scored on reconstruction error over the window, and on REPLAN JITTER: how far apart two
consecutive chunks put the same piece of trajectory. That second number is what a servo feels.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python sliding_window.py --episodes 20
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
    out: pathlib.Path = ROOT / "out" / "sliding_window.json"
    episodes: int = 20
    horizons: tuple[int, ...] = (30, 50)
    # How far ahead one chunk should reach, in source frames. 50 frames is the 1.7 s the current
    # chunk covers; the larger ones ask what the same budget buys if it is spread further.
    spans: tuple[int, ...] = (50, 100, 200, 400)
    stride: int = 10  # observation frames between the windows sampled for the statistics

def window_points(q: np.ndarray, lo: int, hi: int, h: int, mode: str, global_idx: np.ndarray) -> np.ndarray:
    if mode == "per-window":
        return lo + kb.by_curvature_of(q[lo : hi + 1], h)
    ahead = global_idx[global_idx > lo]
    return ahead[:h] if len(ahead) >= h else np.append(ahead, np.full(h - len(ahead), float(hi)))

def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]

    rows = {}
    for h in args.horizons:
        for span in args.spans:
            for mode in ("per-window", "global"):
                err, jit = [], []
                for path in picks:
                    q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
                    if len(q) < span + args.stride + 2:
                        continue
                    truth = kb.batched_cup(chain, q)
                    # Global grid sized so a `span`-long window contains about h of its points.
                    total = max(int(round(h * (len(q) - 1) / span)), h)
                    gidx = kb.by_curvature_of(q, total)
                    prev = None
                    for lo in range(0, len(q) - span - 1, args.stride):
                        hi = lo + span
                        idx = window_points(q, lo, hi, h, mode, gidx)
                        anchor = np.unique(np.concatenate([[float(lo)], idx, [float(hi)]]))
                        rec = kb.reconstruct(q, anchor)[lo : hi + 1]
                        err.append(np.percentile(np.linalg.norm(kb.batched_cup(chain, rec) - truth[lo : hi + 1], axis=1) * 1000, 95))
                        if prev is not None:
                            # Overlap of this chunk with the previous one, compared where both speak.
                            a, b = prev[1][args.stride :], kb.batched_cup(chain, rec)[: span + 1 - args.stride]
                            m = min(len(a), len(b))
                            jit.append(np.percentile(np.linalg.norm(a[:m] - b[:m], axis=1) * 1000, 95))
                        prev = (lo, kb.batched_cup(chain, rec))
                rows[(h, span, mode)] = (float(np.median(err)), float(np.median(jit)) if jit else float("nan"))

    print(f"{len(picks)} episodes. cup error over the window and replan jitter, p95, median (mm)\n")
    print(f"{'H':>4} {'span(frames)':>13} {'ahead(s)':>9} {'per-window err':>15} {'jitter':>8} {'global err':>12} {'jitter':>8}")
    for h in args.horizons:
        for span in args.spans:
            pw, gl = rows[(h, span, "per-window")], rows[(h, span, "global")]
            print(f"{h:4d} {span:13d} {span/30:9.1f} {pw[0]:15.2f} {pw[1]:8.2f} {gl[0]:12.2f} {gl[1]:8.2f}")
    pathlib.Path(args.out).write_text(json.dumps(
        {f"h{h}_span{s}_{m}": {"err_mm": rows[(h, s, m)][0], "replan_jitter_mm": rows[(h, s, m)][1]}
         for h in args.horizons for s in args.spans for m in ("per-window", "global")}, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main(tyro.cli(Args))
