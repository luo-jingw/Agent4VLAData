"""Check the built dataset against the acceptance criteria in PLAN.md, over every episode.

Reads only the parquet columns -- no video decoding -- so it runs in about half a minute.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python verify_dataset.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import progress_resample as ar
import numpy as np
import pandas as pd
import provenance
import tyro

ROOT = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v4"
    out: pathlib.Path = ROOT / "out" / "verify.json"
    parked_eps: float = 1e-3  # rad per frame; the same quiet test the lead-in trim uses


@dataclasses.dataclass
class Tally:
    frames: int = 0  # of the recordings, which is what the curve is built on
    built_rows: int = 0  # of the dataset, after the lead-in trim
    trimmed: int = 0
    expected_rows: int = 0
    samples: int = 0
    parked_frames: int = 0
    parked_samples: int = 0
    event_frames: int = 0
    event_samples: int = 0
    dwell_frames: int = 0
    dwell_samples: int = 0


def main(args: Args) -> None:
    report = json.loads((args.dataset_dir / "meta" / "arc_resample.json").read_text())
    cfg = ar.config_from_report(report)
    sources = provenance.resolve(args.dataset_dir)

    tally = Tally()
    speedups: list[float] = []
    seal_measured: list[int] = []
    for path in sorted((args.dataset_dir / "data").glob("chunk-*/*.parquet")):
        source = sources[int(path.stem.split("_")[1])]
        # The recording, not the built rows: the trim drops rows after the curve is built, so
        # recomputing from the built columns would divide the same progress among fewer free frames
        # and disagree with the chunks actually stored. See provenance.py.
        frame = pd.read_parquet(source.parquet, columns=["state", "actions"])
        states = np.stack(frame["state"].to_numpy()).astype(np.float64)
        actions = np.stack(frame["actions"].to_numpy()).astype(np.float64)
        resampled, curve = ar.resample(states, actions, cfg)

        # Where each output sample came from, on the source frame grid.
        index = np.clip(np.rint(resampled.t).astype(int), 0, len(curve.ds) - 1)
        event = curve.approach_mask | curve.dwell_mask
        # Frames where neither the arm nor the command moved. The old test for this compared
        # raw_step against the dead zone, which no frame in the collection has ever satisfied --
        # the cube root lifts even a frozen arm to about six times the threshold -- so it reported
        # 0.0% and passed on every build without measuring anything.
        step = np.maximum(
            np.linalg.norm(np.diff(states[:, ar.JOINT_DIMS], axis=0), axis=1),
            np.linalg.norm(np.diff(actions[:, ar.JOINT_DIMS], axis=0), axis=1),
        )
        parked = step[: len(curve.ds)] <= args.parked_eps

        tally.frames += len(curve.ds)
        tally.built_rows += len(pd.read_parquet(path, columns=["frame_index"]))
        tally.trimmed += source.start
        tally.expected_rows += len(frame) - source.start
        tally.samples += len(index)
        tally.parked_frames += int(parked.sum())
        tally.parked_samples += int(parked[index].sum())
        tally.event_frames += int(event.sum())
        tally.event_samples += int(event[index].sum())
        tally.dwell_frames += int(curve.dwell_mask.sum())
        tally.dwell_samples += int(curve.dwell_mask[index].sum())
        speedups.append(len(curve.ds) / max(len(index), 1))
        seal_measured.extend(w.measured_frames for w in curve.windows)

    def pct(a: int, b: int) -> float:
        return round(a / b * 100, 2)

    parked_ratio = (
        (tally.parked_samples / tally.samples) / (tally.parked_frames / tally.frames)
        if tally.parked_frames else 0.0
    )
    summary = {
        "episodes": len(speedups),
        "frames": tally.frames,
        "built_rows": tally.built_rows,
        "trimmed_rows": tally.trimmed,
        "samples_equivalent": tally.samples,
        "parked_share_frames_pct": pct(tally.parked_frames, tally.frames),
        "parked_share_samples_pct": pct(tally.parked_samples, tally.samples),
        # Below 1 the resampling is denying parked frames their proportional share of the labels.
        "parked_supervision_ratio": round(parked_ratio, 3),
        "event_share_frames_pct": pct(tally.event_frames, tally.frames),
        "event_share_samples_pct": pct(tally.event_samples, tally.samples),
        "dwell_share_frames_pct": pct(tally.dwell_frames, tally.frames),
        "dwell_share_samples_pct": pct(tally.dwell_samples, tally.samples),
        "compression_min": round(min(speedups), 3),
        "compression_median": round(float(np.median(speedups)), 3),
        "compression_max": round(max(speedups), 3),
        "onsets": len(seal_measured),
        "seal_frames_p50": int(np.percentile(seal_measured, 50)),
        "seal_frames_p90": int(np.percentile(seal_measured, 90)),
    }

    checks = [
        ("parked frames get well under their share of the labels", parked_ratio < 0.5),
        # A row count that drifts from the recording minus the trim means the parquet and the video
        # no longer agree on what frame 0 is, which is silent mislabelling rather than a crash.
        ("built rows equal the recording minus the trim", tally.built_rows == tally.expected_rows),
        ("event supervision not compressed", summary["event_share_samples_pct"] >= summary["event_share_frames_pct"]),
        ("dwell supervision not compressed", summary["dwell_share_samples_pct"] >= summary["dwell_share_frames_pct"]),
        # Source frames per action point. Lands under the configured density because a grasp event
        # spends its whole budget over a short stretch; a fixed band would re-fail on every change of
        # span, so the band follows the config the dataset records.
        ("compression consistent with the configured span",
         0.45 * cfg.frames_per_step <= summary["compression_median"] <= 1.05 * cfg.frames_per_step),
        ("episode count matches the report", summary["episodes"] == report["kept_episodes"]),
    ]

    print(json.dumps(summary, indent=2))
    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "checks": dict(checks)}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
