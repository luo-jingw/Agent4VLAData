"""Run the trained policy on frames from the training set and check what it learned to emit.

Two questions, both about the arc-length labels rather than about task success:

  1. Trajectory error.  How far does the predicted chunk sit from the label it was trained on?
     Reported in radians over the 22 joint dims, per step along the horizon, so drift with lookahead
     is visible. This is an open-loop check against the training set -- it bounds how well the model
     fits the labels, and says nothing about real-robot success.
  2. Equidistance.  The labels are uniform in arc length everywhere except inside a grasp event,
     where they are uniform in time instead. If the model learned the representation, its own output
     should show the same signature: nearly constant displacement per step in free motion, and a
     visible slowdown inside the event windows. A model that ignored the reparameterisation would
     emit the raw teleoperated speed profile, which varies several-fold.

The comparison is against the stored `action_chunk` column, so it exercises exactly the array the
loss was computed on.

  cd tmp/tmp_train_G2
  CUDA_VISIBLE_DEVICES=0 ../../train/openpi/.venv/bin/python closed_loop_check.py --n-samples 100
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import progress_resample as ar
import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "openpi_patch"))

import config_arc  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openpi.policies.policy_config as _policy_config  # noqa: E402
import tyro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
CAMERAS = {
    "observation/image": "image",
    "observation/left_wrist": "left_wrist_image",
    "observation/right_wrist": "right_wrist_image",
}


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v4"
    checkpoint: pathlib.Path = ROOT / "checkpoints" / "g2_arc_v1" / "run1" / "29999"
    out_dir: pathlib.Path = ROOT / "out"
    n_samples: int = 100
    seed: int = 0
    # Fraction of the sample drawn from inside a grasp event window. The events are 15% of the
    # frames but carry the behaviour the whole scheme exists to protect, so they are over-sampled.
    event_fraction: float = 0.4


@dataclasses.dataclass(frozen=True)
class Sample:
    episode: int
    frame: int
    in_event: bool


def read_frame(path: pathlib.Path, index: int) -> np.ndarray:
    """Decode one frame as HWC uint8, seeking on the nearest keyframe."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = int(index / ar.FPS / float(stream.time_base))
        container.seek(target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"{path.name}: frame {index} not reached")


def draw_samples(dataset_dir: pathlib.Path, args: Args) -> list[Sample]:
    """Pick frames, over-sampling the grasp events relative to their share of the timeline."""
    rng = np.random.default_rng(args.seed)
    paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    cfg = ar.config_from_report(json.loads((dataset_dir / "meta" / "arc_resample.json").read_text()))

    event_pool: list[tuple[int, int]] = []
    free_pool: list[tuple[int, int]] = []
    for path in paths:
        episode = int(path.stem.split("_")[1])
        frame = pd.read_parquet(path, columns=["state", "actions"])
        states = np.stack(frame["state"].to_numpy()).astype(np.float64)
        actions = np.stack(frame["actions"].to_numpy()).astype(np.float64)
        curve = ar.progress_curve(states, actions, cfg)
        event = curve.approach_mask | curve.dwell_mask
        for index in np.flatnonzero(event):
            event_pool.append((episode, int(index)))
        for index in np.flatnonzero(~event):
            free_pool.append((episode, int(index)))

    n_event = int(round(args.n_samples * args.event_fraction))
    picks = [
        Sample(*event_pool[i], in_event=True)
        for i in rng.choice(len(event_pool), size=min(n_event, len(event_pool)), replace=False)
    ]
    picks += [
        Sample(*free_pool[i], in_event=False)
        for i in rng.choice(len(free_pool), size=args.n_samples - len(picks), replace=False)
    ]
    return sorted(picks, key=lambda s: (s.episode, s.frame))


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = config_arc.register(ROOT)
    print(f"loading {args.checkpoint}")
    policy = _policy_config.create_trained_policy(config, args.checkpoint)

    tasks = {
        int(json.loads(line)["task_index"]): json.loads(line)["task"]
        for line in (args.dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    }
    samples = draw_samples(args.dataset_dir, args)
    print(f"{len(samples)} samples: {sum(s.in_event for s in samples)} inside grasp events")

    horizon, dim = config_arc.ACTION_HORIZON, config_arc.ACTION_DIM
    joints = list(ar.JOINT_DIMS)
    preds = np.zeros((len(samples), horizon, dim))
    labels = np.zeros((len(samples), horizon, dim))
    in_event = np.zeros(len(samples), dtype=bool)

    cache: dict[int, pd.DataFrame] = {}
    for i, sample in enumerate(samples):
        if sample.episode not in cache:
            cache.clear()  # one episode at a time; the chunk column is 380 MB across the set
            cache[sample.episode] = pd.read_parquet(
                args.dataset_dir / "data" / "chunk-000" / f"episode_{sample.episode:06d}.parquet"
            )
        frame = cache[sample.episode]
        row = frame.iloc[sample.frame]

        obs = {"observation/state": np.asarray(row["state"], dtype=np.float32)}
        for key, video_key in CAMERAS.items():
            obs[key] = read_frame(
                args.dataset_dir / "videos" / "chunk-000" / video_key / f"episode_{sample.episode:06d}.mp4",
                sample.frame,
            )
        obs["prompt"] = tasks[int(row["task_index"])]

        preds[i] = np.asarray(policy.infer(obs)["actions"])[:horizon, :dim]
        labels[i] = np.asarray(row["action_chunk"], dtype=np.float64).reshape(horizon, dim)
        in_event[i] = sample.in_event
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)}")

    # ---- 1. trajectory error -------------------------------------------------------------------
    err = np.linalg.norm(preds[:, :, joints] - labels[:, :, joints], axis=2)  # (n, horizon) rad
    per_dim = np.abs(preds[:, :, joints] - labels[:, :, joints]).mean(axis=(0, 1))
    suction_pred = (preds[:, :, list(ar.SUCTION_ACTION_DIMS)] > 0.5).astype(int)
    suction_label = (labels[:, :, list(ar.SUCTION_ACTION_DIMS)] > 0.5).astype(int)

    # ---- 2. equidistance -----------------------------------------------------------------------
    def step_norm(chunk: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.diff(chunk[:, :, joints], axis=1), axis=2)  # (n, horizon-1)

    pred_step, label_step = step_norm(preds), step_norm(labels)
    STILL = 0.002  # the dead-zone threshold, reused here as "not moving"
    cv = lambda x: float(x.std(axis=1).mean() / x.mean())  # noqa: E731

    summary = {
        "checkpoint": str(args.checkpoint),
        "samples": len(samples),
        "samples_in_event": int(in_event.sum()),
        "trajectory_error_rad": {
            "overall_p50": round(float(np.percentile(err, 50)), 5),
            "overall_p95": round(float(np.percentile(err, 95)), 5),
            "overall_max": round(float(err.max()), 5),
            "step_1": round(float(err[:, 0].mean()), 5),
            "step_25": round(float(err[:, 24].mean()), 5),
            "step_50": round(float(err[:, -1].mean()), 5),
            "worst_joint": {
                "index": joints[int(np.argmax(per_dim))],
                "mean_abs_rad": round(float(per_dim.max()), 5),
            },
        },
        "suction_agreement_pct": round(float((suction_pred == suction_label).mean() * 100), 2),
        "equidistance": {
            "label_step_cv": round(cv(label_step), 4),
            "pred_step_cv": round(cv(pred_step), 4),
            "label_step_mean": round(float(label_step.mean()), 5),
            "pred_step_mean": round(float(pred_step.mean()), 5),
            # Inside an event the labels deliberately slow down; the ratio should be well below 1.
            "label_event_over_free": round(
                float(label_step[in_event].mean() / label_step[~in_event].mean()), 3
            ),
            "pred_event_over_free": round(
                float(pred_step[in_event].mean() / pred_step[~in_event].mean()), 3
            ),
        },
        # During the vacuum build the arm is meant to hold dead still, and the labels contain exact
        # zeros there. A model that instead creeps forward could peel the cup off mid-seal, so the
        # share of near-zero steps is reported separately rather than being averaged away.
        "dead_still_steps_pct": {
            "label_all": round(float((label_step < STILL).mean() * 100), 2),
            "pred_all": round(float((pred_step < STILL).mean() * 100), 2),
            "label_in_event": round(float((label_step[in_event] < STILL).mean() * 100), 2),
            "pred_in_event": round(float((pred_step[in_event] < STILL).mean() * 100), 2),
            "threshold_rad_per_step": STILL,
        },
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / "closed_loop.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.out_dir / "closed_loop_arrays.npz",
        preds=preds.astype(np.float32), labels=labels.astype(np.float32), in_event=in_event,
        episodes=np.array([s.episode for s in samples]), frames=np.array([s.frame for s in samples]),
    )

    # ---- figures -------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(np.arange(1, horizon + 1), np.percentile(err, 50, axis=0), color="#1b9e77", label="p50")
    ax.fill_between(
        np.arange(1, horizon + 1), np.percentile(err, 25, axis=0), np.percentile(err, 75, axis=0),
        color="#1b9e77", alpha=0.25, label="p25-p75",
    )
    ax.plot(np.arange(1, horizon + 1), np.percentile(err, 95, axis=0), color="#d95f02", lw=1, label="p95")
    ax.set_title("trajectory error vs the label, along the horizon")
    ax.set_xlabel("step within the chunk")
    ax.set_ylabel("joint-space error (rad)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, data, colour in (("label", label_step, "0.45"), ("prediction", pred_step, "#1b9e77")):
        ax.plot(np.arange(1, horizon), data.mean(axis=0), color=colour, label=name)
        ax.fill_between(
            np.arange(1, horizon), np.percentile(data, 25, axis=0), np.percentile(data, 75, axis=0),
            color=colour, alpha=0.2,
        )
    ax.set_title("displacement per step: flat = equidistant")
    ax.set_xlabel("step within the chunk")
    ax.set_ylabel("rad / step")
    ax.legend(fontsize=8)

    ax = axes[2]
    bins = np.linspace(0, np.percentile(label_step, 99), 50)
    ax.hist(label_step.ravel(), bins=bins, alpha=0.55, color="0.45", label=f"label (CV {cv(label_step):.2f})")
    ax.hist(pred_step.ravel(), bins=bins, alpha=0.6, color="#1b9e77", label=f"prediction (CV {cv(pred_step):.2f})")
    ax.set_title("displacement distribution")
    ax.set_xlabel("rad / step")
    ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "closed_loop.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out_dir / 'closed_loop.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
