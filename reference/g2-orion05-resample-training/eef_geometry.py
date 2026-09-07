"""Phase 0: does "operator adjustment" exist as a separable geometric phenomenon, and how big is it?

The claim under test is that teleoperated demonstrations contain stretches where the operator is
re-aiming rather than working, that these get learned along with the task, and that they can be
spotted from trajectory geometry alone. Everything is measured at the right-hand suction cup in
Cartesian space, via the same URDF chain `eef_error.py` walks -- "physical constraint" is a statement
about the world, not about joint angles, and millimetres are checkable against the 7.3 mm contact
error the policy already has.

Tortuosity -- path length over net displacement -- is the discriminator, because speed is not one. A
slow committed approach and a slow hunt look identical to any speed or curvature test; only the
first one goes somewhere. But tortuosity alone is not enough either: a frame where the arm is simply
parked has no path AND no displacement, and the ratio of two near-zeros is noise that reads as
enormous. The first pass here made exactly that mistake and reported a fifth of the recording as
"hunting" at a median speed of 0.0 mm/s. Hence the three-way split, gated on path length first:

    still    path over the window below `still_floor_mm`  -- parked; a pause, not an adjustment
    hunt     path above the floor, tortuosity above `tau_hunt`   -- moving, going nowhere
    transit  path above the floor, tortuosity below it          -- moving, going somewhere

Only `hunt` is the phenomenon under test. The 2-D path-by-tortuosity table exists so that this
split can be checked rather than trusted: an adjustment population would show as mass at moderate
path length and high tortuosity, and if that cell is empty at every threshold, it is empty.

What the arc-length pipeline does with each class is reported alongside, since a phase that is
already suppressed by the resampling costs nothing regardless of how much of the recording it fills.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python eef_geometry.py --episodes 10
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import progress_resample as ar
import eef_error as fk
import numpy as np
import pandas as pd
import tyro

ROOT = pathlib.Path(__file__).resolve().parent
JOINTS = tuple(range(22))
RIGHT_CMD_DIM = 23  # the right suction command; the pick task is right-handed throughout

PATH_BINS = (0.0, 1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 400.0, np.inf)
TAU_BINS = (0.0, 1.1, 1.3, 2.0, 3.0, 10.0, np.inf)


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pick-arc-v4"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out: pathlib.Path = ROOT / "out" / "eef_geometry.json"
    episodes: int = 10
    window: int = 15  # +-0.5 s at 30 fps, the span a single adjustment would occupy
    still_floor_mm: float = 5.0  # path over the whole window; below this the arm is parked
    tau_hunt: float = 2.0  # path length twice the net displacement
    voxel_mm: float = 50.0
    label_stride: int = 15  # frames between the sampled chunks in the label check (FK is the cost)


def tip_path(chain: list[fk.Link], poses: np.ndarray) -> np.ndarray:
    """Cartesian tool positions in millimetres for an (n, 22) sequence of joint vectors."""
    return 1000.0 * np.array(
        [fk.forward(chain, {fk.DIM_TO_JOINT[d]: float(pose[d]) for d in JOINTS}) for pose in poses]
    )


def geometry(path: np.ndarray, w: int) -> dict[str, np.ndarray]:
    """Per-frame local shape, each quantity taken over a window centred on the frame."""
    n = len(path)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    path_mm = cumulative[hi] - cumulative[lo]
    net_mm = np.linalg.norm(path[hi] - path[lo], axis=1)
    span = np.maximum(hi - lo, 1)
    return {
        "path_mm": path_mm,
        "net_mm": net_mm,
        "tau": path_mm / np.maximum(net_mm, 1e-3),
        "speed_mms": path_mm / span * ar.FPS,
    }


def classify(feats: dict[str, np.ndarray], floor: float, tau_hunt: float) -> dict[str, np.ndarray]:
    """Split frames into parked / hunting / productive. Gated on path length before tortuosity."""
    still = feats["path_mm"] < floor
    hunt = ~still & (feats["tau"] > tau_hunt)
    return {"still": still, "hunt": hunt, "transit": ~still & ~hunt}


def resample_density(curve: ar.ProgressCurve, n: int) -> np.ndarray:
    """How many resampled action points land on each source frame -- the supervision density."""
    targets = np.arange(0.0, curve.total, curve.step)
    return np.histogram(ar.invert(curve, targets), bins=np.arange(n + 1))[0].astype(float)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC. 0.5 is chance; below 0.5 means the feature is inverted, not useless."""
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    cfg = ar.config_from_report(json.loads((args.dataset_dir / "meta" / "arc_resample.json").read_text()))
    files = sorted((args.dataset_dir / "data").glob("chunk-*/*.parquet"))[: args.episodes]
    print(f"{args.dataset_dir.name}: {len(files)} episodes, right tool, window +-{args.window} frames")

    rows, lead, tail, label_gap, label_net = [], [], [], {"still": [], "moving": []}, {"still": [], "moving": []}
    for episode, path_file in enumerate(files):
        frame = pd.read_parquet(path_file, columns=["state", "actions", "action_chunk"])
        states = np.stack(frame["state"].to_numpy()).astype(np.float64)
        actions = np.stack(frame["actions"].to_numpy()).astype(np.float64)
        chunks = np.stack(frame["action_chunk"].to_numpy()).reshape(len(frame), cfg.action_horizon, -1)
        tool = tip_path(chain, actions[:, :22])
        feats = geometry(tool, args.window)
        phase = classify(feats, args.still_floor_mm, args.tau_hunt)

        moving = np.flatnonzero(~phase["still"])
        first = int(moving[0]) if len(moving) else len(tool)
        lead.append(first)
        tail.append(len(tool) - 1 - int(moving[-1]) if len(moving) else 0)

        curve = ar.progress_curve(states, actions, cfg)
        # The masks live on the n-1 grid the progress metric is built on; pad to frame count.
        event = np.zeros(len(tool), dtype=bool)
        marked = curve.dwell_mask | curve.approach_mask
        event[: len(marked)] = marked

        # What the label at a frame actually commands: where the first point sits relative to the
        # robot now (a jump the controller could not execute), and how far the chunk travels.
        for kind, source in (("still", range(0, first, args.label_stride)),
                             ("moving", moving[:: max(len(moving) // 8, 1)] if len(moving) else [])):
            for i in source:
                points = tip_path(chain, chunks[int(i)][:, :22])
                label_gap[kind].append(float(np.linalg.norm(points[0] - tool[int(i)])))
                label_net[kind].append(float(np.linalg.norm(points[-1] - points[0])))

        rows.append({
            "episode": np.full(len(tool), episode), "xyz": tool, "event": event,
            "density": resample_density(curve, len(tool)),
            "lead_s_share": curve.s[min(first, len(curve.s) - 1)] / max(curve.total, 1e-9),
            **feats, **phase,
        })
        print(f"  ep {episode:2d}: {len(tool):4d} frames  still {phase['still'].mean():5.1%}  "
              f"hunt {phase['hunt'].mean():5.1%}  transit {phase['transit'].mean():5.1%}  "
              f"lead-in {first / ar.FPS:.1f}s")

    scalar = {"lead_s_share"}
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0] if k not in scalar | {"xyz"}}
    data["xyz"] = np.concatenate([r["xyz"] for r in rows])
    total = len(data["tau"])
    out: dict = {"episodes": len(files), "frames": total,
                 "still_floor_mm": args.still_floor_mm, "tau_hunt": args.tau_hunt}

    # ---- Q1: how the recording divides, and where the dead time really is --------------------
    print(f"\n[Q1] {total} frames = {total / ar.FPS:.0f} s")
    for name in ("still", "hunt", "transit"):
        mask = data[name]
        print(f"     {name:<8} {int(mask.sum()):>6d} frames {mask.mean():>6.1%}  "
              f"median speed {np.median(data['speed_mms'][mask]) if mask.any() else 0:>6.1f} mm/s")
        out[name] = {"share": round(float(mask.mean()), 4),
                     "median_speed_mms": round(float(np.median(data["speed_mms"][mask])), 2) if mask.any() else None}
    lead, tail = np.array(lead), np.array(tail)
    out["lead_in_frames_p50"], out["trailing_frames_p50"] = float(np.median(lead)), float(np.median(tail))
    out["lead_in_progress_share_p50"] = round(float(np.median([r["lead_s_share"] for r in rows])), 5)
    print(f"     leading still  p50 {np.median(lead):.0f} frames = {np.median(lead) / ar.FPS:.1f} s, "
          f"p90 {np.percentile(lead, 90):.0f} = {np.percentile(lead, 90) / ar.FPS:.1f} s")
    print(f"     trailing still p50 {np.median(tail):.0f} frames = {np.median(tail) / ar.FPS:.1f} s")
    print(f"     the lead-in holds {out['lead_in_progress_share_p50']:.1%} of the episode's progress")
    inside = float((data["hunt"] & data["event"]).sum())
    out["hunt_inside_event"] = round(inside / max(data["hunt"].sum(), 1), 3)
    print(f"     of the hunt frames, {out['hunt_inside_event']:.0%} sit inside a suction event window "
          "-- already handled as an isochronous window, not operator noise")

    # ---- Q1b: is the split an artefact of the thresholds -------------------------------------
    print(f"\n[Q1b] path length (rows, mm over +-{args.window} frames) by tortuosity (cols), % of frames")
    header = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}" for lo, hi in zip(TAU_BINS, TAU_BINS[1:])]
    corner = "path \\ tau"
    print("      " + f"{corner:<10}" + "".join(f"{h:>9}" for h in header) + f"{'row':>9}")
    table = {}
    for lo, hi in zip(PATH_BINS, PATH_BINS[1:]):
        band = (data["path_mm"] >= lo) & (data["path_mm"] < hi)
        cells = [100 * float((band & (data["tau"] >= t0) & (data["tau"] < t1)).sum()) / total
                 for t0, t1 in zip(TAU_BINS, TAU_BINS[1:])]
        name = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        table[name] = [round(c, 2) for c in cells]
        print(f"      {name:<10}" + "".join(f"{c:>8.1f}%" for c in cells) + f"{100 * band.mean():>8.1f}%")
    out["path_by_tau"] = {"tau_bins": header, "table": table}
    moderate = (data["path_mm"] >= 20) & (data["path_mm"] < 400)
    out["moderate_path_high_tau"] = round(float((moderate & (data["tau"] > args.tau_hunt)).mean()), 4)
    print(f"      an adjustment population would sit at moderate path with high tau: "
          f"{out['moderate_path_high_tau']:.2%} of frames")

    # ---- Q2: is the hunting localised in space -----------------------------------------------
    voxel = np.floor(data["xyz"] / args.voxel_mm).astype(int)
    unique: dict[tuple, int] = {}
    index = np.array([unique.setdefault(tuple(v), len(unique)) for v in voxel])
    counts = np.bincount(index, minlength=len(unique)).astype(float)
    hits = np.bincount(index, weights=data["hunt"].astype(float), minlength=len(unique))
    scores = np.zeros(total)
    for episode in range(len(files)):
        held = data["episode"] == episode
        c = counts - np.bincount(index[held], minlength=len(unique))
        h = hits - np.bincount(index[held], weights=data["hunt"][held].astype(float), minlength=len(unique))
        scores[held] = np.where(c > 0, h / np.maximum(c, 1), data["hunt"].mean())[index[held]]
    loo = auc(scores, data["hunt"])
    out["q2"] = {"voxel_mm": args.voxel_mm, "occupied": int((counts > 0).sum()),
                 "dense": int((counts >= 10).sum()), "loo_auc": round(loo, 3)}
    print(f"\n[Q2] {out['q2']['occupied']} occupied {args.voxel_mm:.0f} mm voxels "
          f"({out['q2']['dense']} with >=10 frames); leave-one-episode-out AUC of the voxel prior "
          f"for hunt frames: {loo:.3f}")

    # ---- Q3: what the arc-length pipeline already does with each phase -----------------------
    print("\n[Q3] supervision density by phase -- share of resampled action points over share of frames")
    out["q3"] = {}
    for name in ("still", "hunt", "transit"):
        mask = data[name]
        points = float(data["density"][mask].sum() / data["density"].sum())
        ratio = points / mask.mean() if mask.mean() > 0 else float("nan")
        out["q3"][name] = {"share_frames": round(float(mask.mean()), 4),
                           "share_points": round(points, 4), "ratio": round(ratio, 3)}
        print(f"     {name:<8} {mask.mean():>6.1%} of frames -> {points:>6.1%} of points   ratio {ratio:>5.2f}x")

    # ---- Q4: what the label at a parked frame actually commands ------------------------------
    print("\n[Q4] the label the model is trained on, in millimetres at the cup")
    out["q4"] = {}
    for kind in ("still", "moving"):
        gap, net = np.array(label_gap[kind]), np.array(label_net[kind])
        out["q4"][kind] = {"first_point_gap_p50": round(float(np.median(gap)), 1),
                           "first_point_gap_p90": round(float(np.percentile(gap, 90)), 1),
                           "chunk_net_p50": round(float(np.median(net)), 1), "n": len(gap)}
        print(f"     at a {kind:<7} frame: first commanded point is {np.median(gap):>5.0f} mm from the "
              f"robot now (p90 {np.percentile(gap, 90):.0f}), and the chunk travels "
              f"{np.median(net):>5.0f} mm over its {cfg.action_horizon} points")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
