"""Same path, three definitions of "evenly spaced": uniform in time, in joint space, or at the cup.

Uniform in TIME is the raw teleoperation -- one sample per recorded frame -- and is the baseline both
of the others are trying to improve on.

The shipped labels put equal JOINT-space distance between consecutive actions. Measured through the
URDF, that buys nothing at the tool: one joint-space unit moves the cup anywhere from 76 to 1100 mm
depending on the arm's configuration, and the labels' per-step Cartesian spread is no better than the
raw recording's. This draws the difference rather than asserting it, and resamples the same episode
under a Cartesian metric so the two can be compared side by side.

The cup path is projected to 2D with PCA, not with t-SNE or UMAP. The question here is whether the
samples are evenly spaced, and t-SNE and UMAP deliberately distort distances -- neighbourhoods are
preserved, spacing is not -- so an even-looking t-SNE plot would say nothing about the metric. PCA is
a linear projection, so a gap on the page is a gap in the data. (The 22-dimensional joint path is a
different matter, and `--joint-embedding` adds a t-SNE panel for it, where a non-linear embedding is
the appropriate tool and no spacing claim is read off it.)

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python uniform_metric_viz.py --episode 0
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import progress_resample as ar
import eef_error as fk
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
JOINTS = list(ar.JOINT_DIMS)


@dataclasses.dataclass(frozen=True)
class Args:
    episode: int = 0
    dataset_dir: pathlib.Path = ROOT / "data" / "task_2-arc-v2"
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    out_dir: pathlib.Path = ROOT / "out"
    # Frames to draw. Defaults to the stretch around the first grasp, which is where the two metrics
    # disagree most: the arm is folded in, so joint motion buys little cup motion.
    span: int = 420


def cup_positions(chain: list[fk.Link], poses: np.ndarray) -> np.ndarray:
    return np.array([fk.forward(chain, {fk.DIM_TO_JOINT[d]: float(p[d]) for d in JOINTS}) for p in poses])


def uniform_samples(path: np.ndarray, n: int) -> np.ndarray:
    """Indices (fractional) of `n` points equally spaced along a polyline's own arc length."""
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    s = s + np.arange(len(s)) * 1e-12  # break ties inside pauses so the grid steps over them
    return np.interp(np.linspace(0.0, s[-1], n), s, np.arange(len(s), dtype=float))


def resample(path: np.ndarray, at: np.ndarray) -> np.ndarray:
    lo = np.clip(np.floor(at).astype(int), 0, len(path) - 2)
    frac = (at - lo)[:, None]
    return path[lo] * (1 - frac) + path[lo + 1] * frac


def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    frame = pd.read_parquet(args.dataset_dir / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    states = np.stack(frame["state"].to_numpy()).astype(np.float64)
    actions = np.stack(frame["actions"].to_numpy()).astype(np.float64)
    report = json.loads((args.dataset_dir / "meta" / "arc_resample.json").read_text())
    curve = ar.progress_curve(states, actions, ar.config_from_report(report))

    onset = curve.windows[0].onset if curve.windows else len(actions) // 2
    lo = max(onset - args.span // 2, 0)
    hi = min(lo + args.span, len(actions) - 1)
    segment = actions[lo:hi]
    cup = cup_positions(chain, segment)

    # Every grid gets the same number of points, so only their placement differs.
    n = 60
    # The grid the shipped labels really use: the dead zone drops pauses, the grasp event runs at
    # recorded time instead of by distance, and the rest is scaled for speed. Reconstructed from the
    # episode's own progress curve rather than re-derived, so it cannot drift from what was trained.
    ship_ds = curve.ds[lo : hi - 1]
    ship_s = np.concatenate([[0.0], np.cumsum(ship_ds)])
    ship_s = ship_s + np.arange(len(ship_s)) * 1e-12
    ship_at = np.interp(np.linspace(0.0, ship_s[-1], n), ship_s, np.arange(len(ship_s), dtype=float))

    grids = {
        "time": (np.linspace(0.0, len(segment) - 1.0, n), "#7570b3", "uniform in TIME\n(raw teleoperation)"),
        "shipped": (ship_at, "#d95f02", "the SHIPPED labels\n(joint arc + dead zone + isochronous grasp)"),
        "joint": (uniform_samples(segment[:, JOINTS], n), "#e7a03c", "plain JOINT arc length\n(no dead zone, no event window)"),
        "cup": (uniform_samples(cup, n), "#1b9e77", "uniform at the CUP\n(the proposal)"),
    }
    grids = {k: (at, resample(cup, at), colour, label) for k, (at, colour, label) in grids.items()}
    ORDER = ("time", "shipped", "joint", "cup")

    mean = cup.mean(axis=0)
    _, sv, vt = np.linalg.svd(cup - mean, full_matrices=False)
    basis, var = vt[:2], (sv**2 / (sv**2).sum())[:2]
    proj = lambda p: (p - mean) @ basis.T  # noqa: E731

    def spacing(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.diff(points, axis=0), axis=1)

    gaps = {name: spacing(pts) for name, (_, pts, _, _) in grids.items()}
    summary = {
        "episode": args.episode,
        "frames": int(hi - lo),
        "samples_each": n,
        "cup_path_length_mm": round(float(spacing(cup).sum()) * 1000, 1),
        "cup_spacing_mm": {
            name: {
                "p05": round(float(np.percentile(g, 5)) * 1000, 2),
                "p50": round(float(np.percentile(g, 50)) * 1000, 2),
                "p95": round(float(np.percentile(g, 95)) * 1000, 2),
                "cv": round(float(g.std() / g.mean()), 3),
            }
            for name, g in gaps.items()
        },
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / "uniform_metric.json").write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(22, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    pp = proj(cup)
    fig.suptitle(
        "One cup path, four ways of choosing where to put the 60 action samples.  "
        "CV = spread of the gaps between neighbouring samples, as a fraction of their average "
        "(0 = perfectly even).",
        fontsize=12,
    )

    for col, name in enumerate(ORDER):
        _, pts, colour, label = grids[name]
        pts = proj(pts)
        title = f"{label}\ncup-spacing CV {summary['cup_spacing_mm'][name]['cv']}"
        ax = fig.add_subplot(gs[0, col])
        ax.plot(pp[:, 0] * 1000, pp[:, 1] * 1000, color="0.8", lw=1.4, zorder=1)
        ax.scatter(pts[:, 0] * 1000, pts[:, 1] * 1000, s=34, color=colour, zorder=3, label=f"{n} samples")
        if curve.windows:
            k = onset - lo
            if 0 <= k < len(pp):
                ax.scatter(pp[k, 0] * 1000, pp[k, 1] * 1000, s=220, marker="*", c="crimson", zorder=4,
                           label="suction fires")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 (mm, {var[0] * 100:.0f}% of the cup path's variance)")
        ax.set_ylabel(f"PC2 (mm, {var[1] * 100:.0f}%)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for name in ORDER:
        g = gaps[name]
        ax.plot(np.arange(1, len(g) + 1), g * 1000, color=grids[name][2], lw=1.3, label=name)
    ax.set_title("A. How far the cup moves between one sample and the next\n"
                 "a flat line means every action step is the same size", fontsize=10)
    ax.set_xlabel("sample number along the path (1 to 59)")
    ax.set_ylabel("mm the cup travels")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(np.percentile(g, 99) for g in gaps.values()) * 1000, 45)
    for name in ORDER:
        ax.hist(gaps[name] * 1000, bins=bins, alpha=0.5, color=grids[name][2], label=name)
    ax.set_title("B. The same numbers as a histogram\n"
                 "one tall narrow bar = even; spread out = uneven", fontsize=10)
    ax.set_xlabel("mm the cup travels between neighbouring samples")
    ax.set_ylabel("how many sample gaps")
    ax.legend(fontsize=8)

    # Where each grid puts its samples on the source timeline: the mechanism behind A and B.
    ax = fig.add_subplot(gs[1, 2])
    rows = ORDER[::-1]
    ax.eventplot([grids[k][0] for k in rows], colors=[grids[k][2] for k in rows],
                 lineoffsets=list(range(len(rows))), linelengths=0.7)
    if curve.windows:
        ax.axvline(onset - lo, color="crimson", lw=1.2, ls="--", label="suction fires")
        ax.legend(fontsize=8)
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels(list(rows))
    ax.set_title("C. Which recorded video frames each scheme picks\n"
                 "each tick is one sample; crowded = many samples on that stretch", fontsize=10)
    ax.set_xlabel("recorded frame number (0 to 420, i.e. 14 seconds)")

    ax = fig.add_subplot(gs[1, 3])
    jq = np.linalg.norm(np.diff(segment[:, JOINTS], axis=0), axis=1)
    cq = np.linalg.norm(np.diff(cup, axis=0), axis=1)
    ratio = cq / np.maximum(jq, 1e-9) * 1000
    ax.plot(ratio, color="0.35", lw=0.9)
    ax.set_yscale("log")
    ax.set_title("D. How much cup motion one unit of joint motion buys\n"
                 "swings 50x, which is why joint-even is not cup-even", fontsize=10)
    ax.set_xlabel("recorded frame number")
    ax.set_ylabel("mm of cup per rad")
    fig.savefig(args.out_dir / f"uniform_metric_ep{args.episode:03d}.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out_dir / f'uniform_metric_ep{args.episode:03d}.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
