"""Verify the arc-length chunks survive the whole path, and look at what they contain.

Three checks, in order of what they would catch:

  1. dataset vs recomputation -- the `action_chunk` column equals a fresh progress_resample run, so
     build_dataset wrote what it claims.
  2. model input vs dataset -- the array the model receives equals the column, element for element,
     after undoing the 32-dim padding. This is the one that catches a silent reshape or a
     LeRobot round-trip that mangles the flat column.
  3. content -- suction commands stay binary, the dwell windows keep one sample per frame, and the
     step profile inside a chunk shows the approach and the dwell where they are meant to be.

Plus a count of how often the neutral prompt is substituted, which should land near the configured
dropout rate.

  cd tmp/tmp_train_G2
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 ../../train/openpi/.venv/bin/python probe_arc_batch.py
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import sys

import progress_resample as ar
import numpy as np
import pandas as pd
import provenance

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "openpi_patch"))

import config_arc  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import tyro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v4"
    episode: int = 0
    batch_size: int = 32
    prompt_samples: int = 2000
    tolerance: float = 1e-5


def dataset_config(dataset_dir: pathlib.Path) -> ar.ResampleConfig:
    """The config this dataset was built with, not today's module defaults."""
    return ar.config_from_report(json.loads((dataset_dir / "meta" / "arc_resample.json").read_text()))


def recording(args: Args, episode: int) -> tuple[np.ndarray, np.ndarray, int]:
    """The full recording behind a built episode, and how many of its rows were trimmed.

    Not the built columns: the lead-in trim drops rows after the curve is built, so recomputing from
    what is stored would produce a different step than the chunks were cut at. See provenance.py.
    """
    source = provenance.resolve(args.dataset_dir)[episode]
    frame = pd.read_parquet(source.parquet, columns=["state", "actions"])
    return (
        np.stack(frame["state"].to_numpy()).astype(np.float64),
        np.stack(frame["actions"].to_numpy()).astype(np.float64),
        source.start,
    )


def check_column_against_recompute(args: Args) -> None:
    frame = pd.read_parquet(args.dataset_dir / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet")
    stored = np.stack(frame["action_chunk"].to_numpy()).reshape(
        len(frame), config_arc.ACTION_HORIZON, config_arc.ACTION_DIM
    )

    states, actions, start = recording(args, args.episode)
    cfg = dataset_config(args.dataset_dir)
    curve = ar.progress_curve(states, actions, cfg)
    fresh = ar.all_chunks(actions, curve, config_arc.ACTION_HORIZON).astype(np.float32)[start:]

    delta = float(np.abs(stored - fresh).max())
    print(f"[1] column vs recompute, episode {args.episode} ({start} lead-in rows trimmed): "
          f"max |delta| = {delta:.3e}", end="")
    print("  OK" if delta <= args.tolerance else "  MISMATCH")

    binary = stored[:, :, list(ar.SUCTION_ACTION_DIMS)]
    print(f"    suction values in the column: {np.unique(binary)}")
    print(f"    step {curve.step:.5f}, dwell frames {int(curve.dwell_mask.sum())}, windows:")
    for w in curve.windows:
        print(
            f"      onset {w.onset:5d}  measured seal {w.measured_frames:3d} frames "
            f"({w.measured_frames / ar.FPS:.2f}s)  used {w.frames:3d}"
        )


def check_model_input_against_column(args: Args) -> None:
    config = dataclasses.replace(config_arc.register(ROOT), batch_size=args.batch_size)
    loader = _data_loader.create_data_loader(config, num_batches=1, shuffle=False, skip_norm_stats=True)
    _, actions = next(iter(loader))
    actions = np.asarray(actions)

    frame = pd.read_parquet(args.dataset_dir / "data" / "chunk-000" / "episode_000000.parquet")
    stored = np.stack(frame["action_chunk"].to_numpy())[: args.batch_size].reshape(
        -1, config_arc.ACTION_HORIZON, config_arc.ACTION_DIM
    )

    print(f"\n[2] model input {actions.shape} vs column {stored.shape}")
    delta = float(np.abs(actions[:, :, : config_arc.ACTION_DIM] - stored).max())
    padding = float(np.abs(actions[:, :, config_arc.ACTION_DIM :]).max())
    print(f"    first {config_arc.ACTION_DIM} dims: max |delta| = {delta:.3e}", end="")
    print("  OK" if delta <= args.tolerance else "  MISMATCH")
    print(f"    padding dims {config_arc.ACTION_DIM}..31 absmax = {padding:.3e}", end="")
    print("  OK" if padding == 0.0 else "  NOT ZERO")


def check_step_profile(args: Args) -> None:
    """Print the within-chunk source-frame spacing around an onset: approach then dwell."""
    states, actions, _ = recording(args, args.episode)
    cfg = dataset_config(args.dataset_dir)
    curve = ar.progress_curve(states, actions, cfg)
    if not curve.windows:
        print("\n[3] no suction onset in this episode")
        return

    onset = curve.windows[0].onset
    start = max(onset - cfg.progress.approach_frames - 20, 0)
    window = curve.windows[0]
    chunk = ar.chunk_at(start, actions, curve, config_arc.ACTION_HORIZON)
    targets = curve.s[start] + np.arange(1, config_arc.ACTION_HORIZON + 1) * curve.step
    t = ar.invert(curve, np.clip(targets, 0.0, curve.total))
    spacing = np.diff(np.concatenate([[float(start)], t]))
    index = np.clip(np.rint(t).astype(int), 0, len(curve.ds) - 1)

    print(f"\n[3] chunk from frame {start}: onset {onset}, sealed {window.sealed}, dwell ends {window.end}")
    budget = cfg.progress.event_points
    print(f"    step {curve.step:.4f}; an event window holds min({budget}, its length) points, so its "
          f"spacing is flat and at least 1.00 source frames per step")
    for label, mask in (("approach", curve.approach_mask), ("dwell", curve.dwell_mask)):
        inside = spacing[mask[index]]
        if len(inside):
            print(
                f"    {label:8s}: {len(inside):2d} chunk steps, spacing "
                f"min {inside.min():.2f} max {inside.max():.2f}"
            )
    free = spacing[~(curve.approach_mask | curve.dwell_mask)[index]]
    if len(free):
        print(f"    free    : {len(free):2d} chunk steps, spacing min {free.min():.2f} max {free.max():.2f}")
    suction = chunk[:, ar.SUCTION_ACTION_DIMS[0] : ar.SUCTION_ACTION_DIMS[1] + 1].max(axis=1)
    print(f"    suction along the chunk: {''.join('1' if v > 0.5 else '.' for v in suction)}")


def check_prompt_dropout(args: Args) -> None:
    dropout = config_arc.PromptDropout(neutral=config_arc.NEUTRAL_PROMPT, probability=config_arc.PROMPT_DROPOUT)
    counts: collections.Counter[str] = collections.Counter()
    for _ in range(args.prompt_samples):
        counts[dropout({"prompt": "episode prompt"})["prompt"]] += 1
    rate = counts[config_arc.NEUTRAL_PROMPT] / args.prompt_samples
    print(f"\n[4] prompt dropout over {args.prompt_samples} draws: {rate:.3f} (configured {config_arc.PROMPT_DROPOUT})")


def main(args: Args) -> None:
    check_column_against_recompute(args)
    check_model_input_against_column(args)
    check_step_profile(args)
    check_prompt_dropout(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
