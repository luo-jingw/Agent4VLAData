"""Does the policy actually know when to fire the suction, and on which arm?

Ten episodes, five per task string, and within each one the frames around every suction onset --
the approach, the trigger, the vacuum dwell and a little after. For each frame the predicted chunk
is compared with the label on the three things that make a grasp work:

  1. Trigger timing.  When the arm is currently off and the label's chunk turns it on at step k, does
     the prediction turn it on at step k too? Error is reported in chunk steps. Because the sampled
     frames march toward the onset, a model that is genuinely tracking the event has its trigger step
     count down with them; one that emits a habitual constant does not.
  2. Arm choice.  The right arm always goes first in this collection and the left follows only on the
     two-workpiece task, so firing the wrong cup is the failure the single-pick prompt exists to
     prevent. Reported per task.
  3. Dwell stillness.  Inside the dwell the labels hold the arm still while the vacuum builds.
     Creeping forward there could peel the cup off mid-seal, so the displacement is compared
     against the label rather than being averaged into the overall error.

  cd tmp/tmp_train_G2
  CUDA_VISIBLE_DEVICES=0 ../../train/openpi/.venv/bin/python suction_check.py
"""

from __future__ import annotations

import collections
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
ARMS = {22: "left", 23: "right"}


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "task_2-arc-v2"
    checkpoint: pathlib.Path = ROOT / "checkpoints" / "g2_arc_v1" / "run1" / "29999"
    out_dir: pathlib.Path = ROOT / "out"
    episodes_per_task: int = 5
    pre_frames: int = 60  # how far before the onset the window starts
    post_frames: int = 20  # how far past the end of the dwell it runs
    stride: int = 4  # sample every Nth frame in the window
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class Probe:
    episode: int
    task_index: int
    frame: int
    onset: int  # the onset this window belongs to
    arm_dim: int
    in_dwell: bool


def read_frame(path: pathlib.Path, index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = int(index / ar.FPS / float(stream.time_base))
        container.seek(target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"{path.name}: frame {index} not reached")


def first_on(chunk_column: np.ndarray) -> int | None:
    """Chunk step (1-based) where a suction channel first reads ON, or None."""
    hits = np.flatnonzero(chunk_column > 0.5)
    return int(hits[0]) + 1 if len(hits) else None


def pick_episodes(dataset_dir: pathlib.Path, args: Args) -> list[tuple[int, int]]:
    """Five episodes per task string, drawn at random."""
    rng = np.random.default_rng(args.seed)
    by_task: dict[int, list[int]] = collections.defaultdict(list)
    for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        index = json.loads(line)["episode_index"]
        frame = pd.read_parquet(
            dataset_dir / "data" / "chunk-000" / f"episode_{index:06d}.parquet", columns=["task_index"]
        )
        by_task[int(frame["task_index"].iloc[0])].append(index)
    out: list[tuple[int, int]] = []
    for task_index, pool in sorted(by_task.items()):
        chosen = rng.choice(pool, size=min(args.episodes_per_task, len(pool)), replace=False)
        out += [(int(e), task_index) for e in sorted(chosen)]
    return out


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = config_arc.register(ROOT)
    policy = _policy_config.create_trained_policy(config, args.checkpoint)
    tasks = {
        int(json.loads(line)["task_index"]): json.loads(line)["task"]
        for line in (args.dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    }

    episodes = pick_episodes(args.dataset_dir, args)
    print("episodes:", {t: [e for e, tt in episodes if tt == t] for t in sorted({t for _, t in episodes})})

    horizon, dim = config_arc.ACTION_HORIZON, config_arc.ACTION_DIM
    joints = list(ar.JOINT_DIMS)
    cfg = ar.ResampleConfig(action_horizon=horizon)
    rows: list[dict] = []
    dwell_steps: dict[int, list[float]] = {0: [], 1: []}
    dwell_steps_label: dict[int, list[float]] = {0: [], 1: []}

    for episode, task_index in episodes:
        frame = pd.read_parquet(args.dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        states = np.stack(frame["state"].to_numpy()).astype(np.float64)
        actions = np.stack(frame["actions"].to_numpy()).astype(np.float64)
        curve = ar.progress_curve(states, actions, cfg)

        probes: list[Probe] = []
        for window in curve.windows:
            lo = max(window.onset - args.pre_frames, 0)
            hi = min(window.end + args.post_frames, len(frame) - 1)
            for index in range(lo, hi, args.stride):
                probes.append(
                    Probe(
                        episode=episode,
                        task_index=task_index,
                        frame=index,
                        onset=window.onset,
                        arm_dim=window.action_dim,
                        in_dwell=bool(curve.dwell_mask[min(index, len(curve.dwell_mask) - 1)]),
                    )
                )

        for probe in probes:
            row = frame.iloc[probe.frame]
            obs = {"observation/state": np.asarray(row["state"], dtype=np.float32)}
            for key, video_key in CAMERAS.items():
                obs[key] = read_frame(
                    args.dataset_dir / "videos" / "chunk-000" / video_key / f"episode_{episode:06d}.mp4",
                    probe.frame,
                )
            obs["prompt"] = tasks[task_index]

            pred = np.asarray(policy.infer(obs)["actions"])[:horizon, :dim]
            label = np.asarray(row["action_chunk"], dtype=np.float64).reshape(horizon, dim)

            entry = {
                "episode": episode,
                "task_index": task_index,
                "frame": probe.frame,
                "onset": probe.onset,
                "frames_to_onset": probe.onset - probe.frame,
                "arm": ARMS[probe.arm_dim],
                "in_dwell": probe.in_dwell,
            }
            for arm_dim, name in ARMS.items():
                currently_on = float(row["state"][22 if arm_dim == 22 else 25]) > 0.5
                entry[f"{name}_on_now"] = bool(currently_on)
                entry[f"{name}_label_step"] = None if currently_on else first_on(label[:, arm_dim])
                entry[f"{name}_pred_step"] = None if currently_on else first_on(pred[:, arm_dim])
            rows.append(entry)

            if probe.in_dwell:
                dwell_steps[task_index].append(
                    float(np.linalg.norm(np.diff(pred[:, joints], axis=0), axis=1).mean())
                )
                dwell_steps_label[task_index].append(
                    float(np.linalg.norm(np.diff(label[:, joints], axis=0), axis=1).mean())
                )
        print(f"  episode {episode} (task {task_index}): {len(probes)} frames")

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "suction_check.csv", index=False)

    summary: dict = {"checkpoint": str(args.checkpoint), "frames": len(table), "per_task": {}}
    for task_index, group in table.groupby("task_index"):
        stats: dict = {
            "episodes": sorted(group["episode"].unique().tolist()),
            "frames": len(group),
        }
        for name in ARMS.values():
            off = group[~group[f"{name}_on_now"]]
            has_label = off[off[f"{name}_label_step"].notna()]
            has_pred = off[off[f"{name}_pred_step"].notna()]
            both = off[off[f"{name}_label_step"].notna() & off[f"{name}_pred_step"].notna()]
            delta = (both[f"{name}_pred_step"] - both[f"{name}_label_step"]).astype(float)
            stats[name] = {
                # Of the frames whose label chunk contains a trigger, how many did the model fire in?
                "recall_pct": round(len(both) / max(len(has_label), 1) * 100, 1) if len(has_label) else None,
                # Frames where the model fired but the label did not: a premature or phantom grasp.
                "false_fire": int(len(has_pred) - len(both)),
                "n_compared": len(both),
                "step_error_p50": round(float(delta.median()), 1) if len(delta) else None,
                "step_error_p90": round(float(delta.abs().quantile(0.9)), 1) if len(delta) else None,
                "within_5_steps_pct": round(float((delta.abs() <= 5).mean() * 100), 1) if len(delta) else None,
            }
        if dwell_steps[task_index]:
            stats["dwell_step_rad"] = {
                "label": round(float(np.mean(dwell_steps_label[task_index])), 5),
                "pred": round(float(np.mean(dwell_steps[task_index])), 5),
            }
        summary["per_task"][int(task_index)] = stats

    print(json.dumps(summary, indent=2))
    (args.out_dir / "suction_check.json").write_text(json.dumps(summary, indent=2))

    # ---- figure --------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, task_index in zip(axes, sorted(summary["per_task"]), strict=False):
        group = table[table["task_index"] == task_index]
        for name, colour in (("right", "#1b9e77"), ("left", "#7570b3")):
            sub = group[group[f"{name}_label_step"].notna() & group[f"{name}_pred_step"].notna()]
            if not len(sub):
                continue
            ax.scatter(sub[f"{name}_label_step"], sub[f"{name}_pred_step"], s=26, alpha=0.6,
                       color=colour, label=f"{name} cup ({len(sub)})")
        lim = [0, config_arc.ACTION_HORIZON]
        ax.plot(lim, lim, color="0.4", ls="--", lw=1, label="perfect timing")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"task {task_index}: {'two workpieces' if task_index == 0 else 'one workpiece'}")
        ax.set_xlabel("label: trigger at chunk step")
        ax.set_ylabel("prediction: trigger at chunk step")
        ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "suction_check.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out_dir / 'suction_check.png'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
