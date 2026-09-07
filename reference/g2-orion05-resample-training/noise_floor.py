"""Separate representation loss from jitter the reconstruction was right not to reproduce.

Every number in the keypoint study is a distance to the RAW recording, and the raw recording is not
smooth: teleoperation at 30 fps carries operator tremor, sensor noise and quantisation. A
reconstruction built from a handful of keypoints is smooth by construction, so part of its "error" is
simply the jitter it declined to chase -- and chasing it would be wrong, since the robot cannot
execute it and the controller would filter it out anyway.

So this measures the floor. Smooth the recording lightly, and the distance between raw and smoothed
is the jitter budget: any reconstruction sitting at that level is already as close as a smooth curve
can meaningfully get. Reconstruction error is then reported against both references, raw and smoothed,
and the two together say whether a given N is limited by the representation or by the noise.

The second half asks what the controller is left holding. Executing sparse keypoints at 30 Hz needs
them resampled, and the resulting command stream is compared with the recording on acceleration and
jerk -- the quantities a servo has to absorb.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python noise_floor.py --episodes 20
"""
from __future__ import annotations
import dataclasses, json, pathlib
import progress_resample as ar, eef_error as fk, keypoint_budget as kb
import numpy as np, pandas as pd, scipy.signal, tyro

ROOT = pathlib.Path(__file__).resolve().parent
J = list(ar.JOINT_DIMS)

@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v1"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "noise_floor.json"
    episodes: int = 20
    counts: tuple[int, ...] = (20, 30, 40, 60, 80, 120, 200)
    # Light enough to leave the motion intact: 9 frames is 0.3 s, and a reach lasts several seconds.
    savgol_window: int = 9
    savgol_poly: int = 3

def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    paths = sorted((args.dataset_dir / "data" / "chunk-000").glob("*.parquet"))
    picks = paths[:: max(len(paths) // args.episodes, 1)][: args.episodes]

    floor, raw_acc, raw_jerk = [], [], []
    err_raw = {r: {n: [] for n in args.counts} for r in ("curvature", "greedy")}
    err_smooth = {r: {n: [] for n in args.counts} for r in ("curvature", "greedy")}
    rec_acc = {r: {n: [] for n in args.counts} for r in ("curvature", "greedy")}

    for path in picks:
        q = np.stack(pd.read_parquet(path, columns=["actions"])["actions"].to_numpy()).astype(np.float64)[:, J]
        qs = scipy.signal.savgol_filter(q, args.savgol_window, args.savgol_poly, axis=0)
        cup_raw, cup_smooth = kb.batched_cup(chain, q), kb.batched_cup(chain, qs)
        floor.append(np.percentile(np.linalg.norm(cup_raw - cup_smooth, axis=1) * 1000, 95))
        raw_acc.append(np.sqrt((np.diff(cup_raw, n=2, axis=0) ** 2).sum(axis=1).mean()) * 1000)
        raw_jerk.append(np.sqrt((np.diff(cup_raw, n=3, axis=0) ** 2).sum(axis=1).mean()) * 1000)

        order = kb.greedy_order_eef(q, cup_raw, chain, max(args.counts))
        for rule in ("curvature", "greedy"):
            for n in args.counts:
                idx = kb.by_curvature_of(q, n) if rule == "curvature" else np.array(sorted(order[:n]), float)
                rec = kb.batched_cup(chain, kb.reconstruct(q, idx))
                err_raw[rule][n].append(np.percentile(np.linalg.norm(rec - cup_raw, axis=1) * 1000, 95))
                err_smooth[rule][n].append(np.percentile(np.linalg.norm(rec - cup_smooth, axis=1) * 1000, 95))
                rec_acc[rule][n].append(np.sqrt((np.diff(rec, n=2, axis=0) ** 2).sum(axis=1).mean()) * 1000)

    med = lambda x: round(float(np.median(x)), 3)  # noqa: E731
    summary = {
        "episodes": len(picks),
        "jitter_floor_mm_p95": med(floor),
        "recording_cup_accel_mm_per_frame2": med(raw_acc),
        "recording_cup_jerk_mm_per_frame3": med(raw_jerk),
        "per_rule": {r: {n: {"vs_raw_mm": med(err_raw[r][n]), "vs_smoothed_mm": med(err_smooth[r][n]),
                             "recon_accel_mm_per_frame2": med(rec_acc[r][n])} for n in args.counts}
                     for r in err_raw},
    }
    print(f"{len(picks)} episodes\n")
    print(f"jitter floor (raw vs lightly smoothed, cup p95): {summary['jitter_floor_mm_p95']:.2f} mm")
    print(f"  -> any reconstruction at or below this is as close as a smooth curve can get")
    print(f"recording cup acceleration {summary['recording_cup_accel_mm_per_frame2']:.3f}  "
          f"jerk {summary['recording_cup_jerk_mm_per_frame3']:.3f}  (mm per frame^2 / frame^3)\n")
    for rule in err_raw:
        print(f"{rule}:")
        print(f"  {'N':>5} {'vs raw':>9} {'vs smoothed':>12} {'noise-limited?':>15} {'recon accel':>12}")
        for n in args.counts:
            row = summary["per_rule"][rule][n]
            limited = "yes" if row["vs_smoothed_mm"] < summary["jitter_floor_mm_p95"] else "no"
            print(f"  {n:5d} {row['vs_raw_mm']:9.2f} {row['vs_smoothed_mm']:12.2f} {limited:>15} "
                  f"{row['recon_accel_mm_per_frame2']:12.3f}")
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main(tyro.cli(Args))
