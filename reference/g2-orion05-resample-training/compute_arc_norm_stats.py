"""Normalization stats for the arc-resampled G2 dataset, read straight from the parquet columns.

Used instead of openpi's `scripts/compute_norm_stats.py`, which routes the dataset through the full
transform pipeline and therefore decodes every video frame -- about an hour here -- while only ever
reading `state` and `actions`. AgiBotG2Inputs passes both through unchanged, so the parquet columns
are exactly what the model sees.

Two differences from the pack's version:

  * Action stats are computed over `action_chunk`, not over the per-frame `actions` column, because
    the chunk is what the model is trained to emit. The two are close but not identical: the chunk
    holds PCHIP samples taken between recorded frames.
  * Suction dims are pinned to [0, 1]. They are binary commands, and pinning guarantees they map to
    exactly {-1, +1} after quantile normalization no matter how rarely an arm is used. pi0.5
    discretizes the normalized state into 256 bins written into the prompt, so a bounded range
    matters for the state as much as for the action targets.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python compute_arc_norm_stats.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "openpi_patch"))

import config_arc  # noqa: E402
import openpi.shared.normalize as normalize  # noqa: E402
import tyro  # noqa: E402

STATE_SUCTION_DIMS = (22, 25)  # left_suction_enabled, right_suction_enabled
ACTION_SUCTION_DIMS = (22, 23)  # left_suction_cmd, right_suction_cmd

STATE_NAMES = [
    "ankle_pitch", "knee_pitch", "hip_pitch", "hip_roll", "hip_yaw",
    "head_yaw", "head_roll", "head_pitch",
    "L_arm1", "L_arm2", "L_arm3", "L_arm4", "L_arm5", "L_arm6", "L_arm7",
    "R_arm1", "R_arm2", "R_arm3", "R_arm4", "R_arm5", "R_arm6", "R_arm7",
    "L_suction", "L_vac_ch2", "L_vac_ch1", "R_suction", "R_vac_ch2", "R_vac_ch1",
]  # fmt: skip
ACTION_NAMES = [*STATE_NAMES[:22], "L_suction_cmd", "R_suction_cmd"]

Scheme = Literal["quantile", "minmax"]


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent / "data" / "pickplace-arc-v4"
    out_dir: pathlib.Path | None = None  # default: the config's own assets dir
    scheme: Scheme = "quantile"


def load_columns(dataset_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_dir / 'data'}")
    states, chunks = [], []
    for path in files:
        frame = pd.read_parquet(path, columns=["state", "action_chunk"])
        states.append(np.stack(frame["state"].to_numpy()))
        chunks.append(np.stack(frame["action_chunk"].to_numpy()))
    state = np.concatenate(states).astype(np.float64)
    chunk = np.concatenate(chunks).astype(np.float64)
    return state, chunk.reshape(-1, config_arc.ACTION_DIM)


def make_stats(x: np.ndarray, scheme: Scheme, suction_dims: tuple[int, ...]) -> normalize.NormStats:
    if scheme == "minmax":
        q01, q99 = x.min(0).copy(), x.max(0).copy()
    else:
        q01, q99 = np.quantile(x, 0.01, 0).copy(), np.quantile(x, 0.99, 0).copy()
    for dim in suction_dims:
        q01[dim], q99[dim] = 0.0, 1.0
    return normalize.NormStats(
        mean=x.mean(0).astype(np.float32),
        std=x.std(0).astype(np.float32),
        q01=q01.astype(np.float32),
        q99=q99.astype(np.float32),
    )


def report(label: str, x: np.ndarray, stats: normalize.NormStats, names: list[str]) -> list[str]:
    q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
    span = q99 - q01
    normalized = 2 * (x - q01) / (span + 1e-6) - 1
    degenerate = [n for n, sp in zip(names, span, strict=True) if sp == 0]

    print(f"\n{label}: {x.shape[0]} rows, {x.shape[1]} dims")
    print(f"  {'i':>2} {'name':<12} {'min':>8} {'max':>8} {'q01':>8} {'q99':>8} {'span':>8} {'norm absmax':>12}")
    for i, name in enumerate(names):
        note = "  DEGENERATE (span=0)" if span[i] == 0 else ""
        print(
            f"  {i:>2} {name:<12} {x[:, i].min():>8.3f} {x[:, i].max():>8.3f} {q01[i]:>8.3f} "
            f"{q99[i]:>8.3f} {span[i]:>8.3f} {np.abs(normalized[:, i]).max():>12.3f}{note}"
        )
    print(f"  overall normalized absmax: {np.abs(normalized).max():.3f}")
    print(f"  non-finite values: {int((~np.isfinite(normalized)).sum())}")
    return [f"{label}.{n}" for n in degenerate]


def main(args: Args) -> None:
    state, actions = load_columns(args.dataset_dir)
    stats = {
        "state": make_stats(state, args.scheme, STATE_SUCTION_DIMS),
        "actions": make_stats(actions, args.scheme, ACTION_SUCTION_DIMS),
    }
    print(f"{args.dataset_dir}  scheme={args.scheme}")
    degenerate = report("state", state, stats["state"], STATE_NAMES)
    degenerate += report("actions", actions, stats["actions"], ACTION_NAMES)

    out = args.out_dir or (
        pathlib.Path(__file__).resolve().parent / "assets" / config_arc.CONFIG_NAME / config_arc.REPO_ID
    )
    normalize.save(out, stats)
    print(f"\nwrote {out / 'norm_stats.json'}")
    if degenerate:
        print(
            f"WARNING: zero-span dimensions: {', '.join(degenerate)}. They are constant in this "
            "collection; the +1e-6 epsilon keeps them finite, normalising to a constant -1."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
