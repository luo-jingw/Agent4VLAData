"""Does uniform spacing in joint space give uniform speed at the suction cup?

The resampling makes each action step cover the same distance in JOINT space. That is not the same as
the same distance in CARTESIAN space: the Jacobian changes with the arm's configuration, so a fixed
joint-space step moves the cup far when the arm is extended and little when it is folded. If the
Cartesian spread turns out to be much wider than the joint-space spread, then joint-space arc length
is a poor stand-in for what the task actually cares about -- and now that the URDF is in hand, the
metric could simply be measured at the tool instead.

Three sequences are compared, all through the same forward kinematics:

  recording   the raw 30 fps frames        uniform in time
  label       the arc-resampled chunk      uniform in joint-space distance by construction
  prediction  what the policy emits        whatever it learned

For each, the per-step displacement of the cup is reported, and the coefficient of variation is the
number to read: 0 would be a perfectly constant tool speed.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python eef_speed.py
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
JOINTS = list(ar.JOINT_DIMS)


@dataclasses.dataclass(frozen=True)
class Args:
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    arrays: pathlib.Path = ROOT / "out" / "closed_loop_arrays.npz"
    dataset_dir: pathlib.Path = ROOT / "data" / "task_2-arc-v2"
    out: pathlib.Path = ROOT / "out" / "eef_speed.json"
    recording_episodes: int = 12


def tip_path(chain: list[fk.Link], poses: np.ndarray) -> np.ndarray:
    """Cartesian positions of the tool for a (n, 22) sequence of joint vectors."""
    return np.array(
        [fk.forward(chain, {fk.DIM_TO_JOINT[d]: float(pose[d]) for d in JOINTS}) for pose in poses]
    )


def spread(steps: np.ndarray) -> dict[str, float]:
    steps = steps[steps > 0] if (steps > 0).any() else steps
    return {
        "mean_mm": round(float(steps.mean()) * 1000, 3),
        "p50_mm": round(float(np.percentile(steps, 50)) * 1000, 3),
        "p05_mm": round(float(np.percentile(steps, 5)) * 1000, 3),
        "p95_mm": round(float(np.percentile(steps, 95)) * 1000, 3),
        "cv": round(float(steps.std() / steps.mean()), 3),
    }


def main(args: Args) -> None:
    chain = fk.build_chain(args.urdf, fk.TIPS["right"])
    data = np.load(args.arrays)
    preds, labels, episodes, frames = data["preds"], data["labels"], data["episodes"], data["frames"]

    label_steps, pred_steps, joint_steps = [], [], []
    for i in range(len(labels)):
        lab = tip_path(chain, labels[i])
        pre = tip_path(chain, preds[i])
        label_steps.append(np.linalg.norm(np.diff(lab, axis=0), axis=1))
        pred_steps.append(np.linalg.norm(np.diff(pre, axis=0), axis=1))
        joint_steps.append(np.linalg.norm(np.diff(labels[i][:, JOINTS], axis=0), axis=1))

    # The time-uniform baseline: what the cup does between consecutive recorded frames.
    recording = []
    for path in sorted((args.dataset_dir / "data").glob("chunk-*/*.parquet"))[: args.recording_episodes]:
        table = pd.read_parquet(path, columns=["actions"])
        actions = np.stack(table["actions"].to_numpy()).astype(np.float64)[::2]  # every 2nd frame
        tip = tip_path(chain, actions)
        recording.append(np.linalg.norm(np.diff(tip, axis=0), axis=1))

    label_cart = np.concatenate(label_steps)
    pred_cart = np.concatenate(pred_steps)
    joint_arc = np.concatenate(joint_steps)
    rec_cart = np.concatenate(recording)

    summary = {
        "note": "right suction cup, per step. Recording is per 2 recorded frames; chunks are per action step.",
        "joint_space_label_rad": {
            "mean": round(float(joint_arc.mean()), 5),
            "cv": round(float(joint_arc.std() / joint_arc.mean()), 3),
        },
        "cartesian_recording_time_uniform": spread(rec_cart),
        "cartesian_label_jointspace_uniform": spread(label_cart),
        "cartesian_prediction": spread(pred_cart),
        # How much Cartesian speed a single joint-space step buys, and how much that ratio moves.
        "mm_per_joint_rad": {
            "p05": round(float(np.percentile(label_cart / np.maximum(joint_arc, 1e-9), 5)) * 1000, 1),
            "p50": round(float(np.percentile(label_cart / np.maximum(joint_arc, 1e-9), 50)) * 1000, 1),
            "p95": round(float(np.percentile(label_cart / np.maximum(joint_arc, 1e-9), 95)) * 1000, 1),
        },
    }
    print(json.dumps(summary, indent=2))
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
