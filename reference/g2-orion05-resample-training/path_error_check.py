"""How far does the predicted chunk stray from the trajectory that was actually recorded?

The earlier check compared the prediction with the arc-length label step by step. That answers "did
the model fit its labels", but it cannot answer "does it follow the demonstrated path", because the
labels are a reparameterisation: at step k the label is deliberately further along than the recording
is at frame k. Matching them index by index against the raw recording would score the intended
speed-up as error.

So this compares the two as CURVES. For every predicted point, the distance measured is to the
recorded polyline itself -- the nearest point on any of its segments, not the recorded point with the
same index. A model that follows the same path faster scores zero here; a model that cuts a corner or
drifts off the path is what the number is meant to catch.

Three curves are scored against the same recording, which is what makes the model's share readable:

  label -> recording    the representation's own error: PCHIP resampling of the same path
  pred  -> recording    label error plus whatever the model adds
  pred  -> label path   the model against the path it was actually trained to emit

Reuses the predictions saved by closed_loop_check.py, so no inference runs here.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python path_error_check.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import progress_resample as ar
import numpy as np
import pandas as pd
import tyro

ROOT = pathlib.Path(__file__).resolve().parent
JOINTS = list(ar.JOINT_DIMS)


@dataclasses.dataclass(frozen=True)
class Args:
    arrays: pathlib.Path = ROOT / "out" / "closed_loop_arrays.npz"
    dataset_dir: pathlib.Path = ROOT / "data" / "task_2-arc-v2"
    out: pathlib.Path = ROOT / "out" / "path_error.json"
    # Extra arc length to keep on the recorded polyline past what the chunk covers, as a fraction.
    # Without a margin the last predicted point has no segment ahead of it to project onto.
    margin: float = 0.25


def point_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest place on a polyline, in the polyline's own units.

    Projects onto every segment rather than snapping to vertices: with the recording at 30 fps and
    the chunk stepping further per sample, vertex-snapping alone would report a sawtooth that is an
    artefact of the recording's own sample spacing.
    """
    starts, ends = polyline[:-1], polyline[1:]
    seg = ends - starts  # (m, d)
    length2 = (seg**2).sum(axis=1)
    length2 = np.where(length2 > 0, length2, 1.0)
    delta = points[:, None, :] - starts[None, :, :]  # (n, m, d)
    t = np.clip((delta * seg[None]).sum(axis=2) / length2[None], 0.0, 1.0)  # (n, m)
    nearest = starts[None] + t[..., None] * seg[None]  # (n, m, d)
    return np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=1)


def recorded_window(actions: np.ndarray, frame: int, arc_needed: float, margin: float) -> np.ndarray:
    """The recorded path from `frame` onward, long enough to cover the chunk plus a margin."""
    step = np.linalg.norm(np.diff(actions[frame:, JOINTS], axis=0), axis=1)
    covered = np.cumsum(step)
    target = arc_needed * (1.0 + margin)
    reach = int(np.searchsorted(covered, target)) + 2 if covered.size and covered[-1] >= target else len(step)
    return actions[frame : frame + reach + 1, JOINTS]


def main(args: Args) -> None:
    data = np.load(args.arrays)
    preds, labels = data["preds"], data["labels"]
    episodes, frames, in_event = data["episodes"], data["frames"], data["in_event"]

    cache: dict[int, np.ndarray] = {}
    rows: list[dict] = []
    for i, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        episode = int(episode)
        if episode not in cache:
            cache.clear()
            table = pd.read_parquet(
                args.dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet",
                columns=["actions"],
            )
            cache[episode] = np.stack(table["actions"].to_numpy()).astype(np.float64)
        actions = cache[episode]

        pred_path = preds[i][:, JOINTS].astype(np.float64)
        label_path = labels[i][:, JOINTS].astype(np.float64)
        arc = float(np.linalg.norm(np.diff(label_path, axis=0), axis=1).sum())
        recording = recorded_window(actions, int(frame), arc, args.margin)
        if len(recording) < 2:
            continue  # too close to the end of the episode to have a path to compare against

        rows.append(
            {
                "in_event": bool(in_event[i]),
                "pred_vs_recording": point_to_polyline(pred_path, recording),
                "label_vs_recording": point_to_polyline(label_path, recording),
                "pred_vs_label_path": point_to_polyline(pred_path, label_path),
                # How much further along the path the chunk gets than the recording does in the same
                # number of 30 Hz ticks: the reparameterisation's intended effect, not an error.
                "arc_ratio": arc
                / max(
                    float(
                        np.linalg.norm(
                            np.diff(actions[int(frame) : int(frame) + len(pred_path) + 1, JOINTS], axis=0), axis=1
                        ).sum()
                    ),
                    1e-9,
                ),
            }
        )

    def stats(key: str, mask: np.ndarray | None = None) -> dict[str, float]:
        pick = rows if mask is None else [r for r, m in zip(rows, mask, strict=True) if m]
        values = np.concatenate([r[key] for r in pick])
        per_sample_max = np.array([r[key].max() for r in pick])
        return {
            "n_samples": len(pick),
            "p50": round(float(np.percentile(values, 50)), 5),
            "p95": round(float(np.percentile(values, 95)), 5),
            "max": round(float(values.max()), 5),
            "per_chunk_max_p50": round(float(np.median(per_sample_max)), 5),
        }

    event = np.array([r["in_event"] for r in rows])
    summary = {
        "samples": len(rows),
        "units": "radians, L2 over the 22 joint dims, distance to the nearest point on the curve",
        "label_vs_recording": stats("label_vs_recording"),
        "pred_vs_recording": stats("pred_vs_recording"),
        "pred_vs_label_path": stats("pred_vs_label_path"),
        "pred_vs_recording_in_event": stats("pred_vs_recording", event),
        "pred_vs_recording_free": stats("pred_vs_recording", ~event),
        "arc_ratio_chunk_over_recording": {
            "p50": round(float(np.median([r["arc_ratio"] for r in rows])), 3),
            "p05": round(float(np.percentile([r["arc_ratio"] for r in rows], 5)), 3),
            "p95": round(float(np.percentile([r["arc_ratio"] for r in rows], 95)), 3),
        },
    }
    print(json.dumps(summary, indent=2))
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
