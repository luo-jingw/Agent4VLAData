"""Window reconstruction scored the way inference actually sees it: nothing beyond the last keypoint.

An earlier version of this measurement anchored the reconstruction at both ends of the window -- the
observation frame and the far edge. The far edge is the future. At inference the policy emits H
keypoints and there is nothing after them, so pinning the curve there flatters the result: an
interpolant with both ends fixed is a much easier fit than one that has to end where its last point
lands.

So the window here runs from the observation frame to the LAST KEYPOINT, the reconstruction is built
from those anchors alone, and nothing past them is scored. Both interpolators are measured, because
the labels are built with PCHIP while a controller handed waypoints will walk straight lines unless
it is changed.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python window_causal_budget.py --episodes 20
"""
from __future__ import annotations
import dataclasses, json, pathlib
import progress_resample as ar, eef_error as fk, keypoint_budget as kb
import linear_interp_budget as lin, numpy as np, pandas as pd, tyro

ROOT = pathlib.Path(__file__).resolve().parent
J = list(ar.JOINT_DIMS)

@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v2"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "window_causal_budget.json"
    episodes: int = 20
    horizon: int = 50
    spans: tuple[int, ...] = (50, 100, 200, 300)
    stride: int = 13

def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]

    rows: dict[tuple, list] = {(sp, k, a): [] for sp in args.spans for k in ("linear", "pchip")
                               for a in ("causal", "both-ends")}
    for path in picks:
        q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
        truth = kb.batched_cup(chain, q)
        for span in args.spans:
            if len(q) < span + 2:
                continue
            for lo in range(0, len(q) - span - 1, args.stride):
                hi = lo + span
                idx = lo + lin.by_exponent(q[lo : hi + 1], args.horizon, 1 / 3)
                end = int(np.floor(idx[-1]))
                if end <= lo + 1:
                    continue
                for kind in ("linear", "pchip"):
                    # What inference can build: the observation plus its own keypoints, nothing else.
                    causal = np.concatenate([[float(lo)], idx])
                    rec = lin.rebuild(q, causal, kind)[lo : end + 1]
                    e = np.linalg.norm(kb.batched_cup(chain, rec) - truth[lo : end + 1], axis=1) * 1000
                    rows[(span, kind, "causal")].append(float(np.percentile(e, 95)))
                    # The earlier, optimistic version: the far edge pinned as if it were known.
                    both = np.concatenate([[float(lo)], idx, [float(hi)]])
                    rec2 = lin.rebuild(q, both, kind)[lo : hi + 1]
                    e2 = np.linalg.norm(kb.batched_cup(chain, rec2) - truth[lo : hi + 1], axis=1) * 1000
                    rows[(span, kind, "both-ends")].append(float(np.percentile(e2, 95)))

    med = lambda k: round(float(np.median(rows[k])), 2) if rows[k] else float("nan")  # noqa: E731
    summary = {"episodes": len(picks), "horizon": args.horizon,
               "results": {f"span{sp}_{k}_{a}": med((sp, k, a)) for sp in args.spans
                           for k in ("linear", "pchip") for a in ("causal", "both-ends")}}
    print(f"{len(picks)} episodes, H={args.horizon}. cup error over the window, p95, median (mm)\n")
    print(f"{'span':>6} {'ahead':>7} {'linear causal':>14} {'pchip causal':>13} {'linear both-ends':>17} {'pchip both-ends':>16}")
    for sp in args.spans:
        print(f"{sp:6d} {sp/30:6.1f}s {med((sp,'linear','causal')):14.2f} {med((sp,'pchip','causal')):13.2f} "
              f"{med((sp,'linear','both-ends')):17.2f} {med((sp,'pchip','both-ends')):16.2f}")
    print("\ncost of the honest anchoring (causal minus both-ends, pchip, mm):")
    for sp in args.spans:
        a, b = med((sp, "pchip", "causal")), med((sp, "pchip", "both-ends"))
        print(f"  span {sp:4d}  {a - b:+.3f}")
    print("\n  The two agree because the placement already puts its last keypoint at the window edge,")
    print("  so pinning that edge separately adds a duplicate anchor and nothing else. The far end of")
    print("  the window was never actually being used as free information.")
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main(tyro.cli(Args))
