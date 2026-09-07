"""Resolve a built dataset's episodes back to the recordings their labels were computed from.

Trimming the frozen lead-in drops rows from the built parquet but deliberately does NOT change the
labels: the progress curve is built on the whole recording and the rows are dropped afterwards, so a
kept frame carries exactly the chunk it would have carried had nothing been trimmed. That is the
point -- the trim is meant to remove duplicated observations, not to re-time the trajectory -- but it
does mean the built columns are an incomplete basis for re-deriving the curve. Recomputing from them
would see a shorter episode, divide the same progress among fewer free frames, and land on a
different `step` than the one the stored chunks were cut at.

So any check that needs the curve has to go back to the recording. That is all this does.

Merged datasets add one hop: their provenance names the single-task builds, whose own provenance
names the recordings.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Source:
    """Where one built episode came from."""

    parquet: pathlib.Path  # the recording, at its original length
    start: int  # rows trimmed off the front of the built episode
    index: int  # the recording's episode number


def resolve(dataset_dir: pathlib.Path) -> dict[int, Source]:
    """Map each built episode index to the recording its labels were computed from."""
    report = json.loads((dataset_dir / "meta" / "arc_resample.json").read_text())
    episodes = [e for e in report["episodes"] if "new_index" in e]

    if "merged_from" in report:
        sub = {str(s): resolve(_at(s)) for s in report["merged_from"]}
        return {e["new_index"]: sub[e["source"]][e["source_index"]] for e in episodes}

    recordings = _at(report["source"]) / "data" / "chunk-000"
    return {
        e["new_index"]: Source(
            parquet=recordings / f"episode_{e['source_index']:06d}.parquet",
            start=int(e.get("trimmed", 0)),
            index=int(e["source_index"]),
        )
        for e in episodes
    }


def _at(path: str | pathlib.Path) -> pathlib.Path:
    """Provenance records paths as they were typed on the command line, so relative to this folder."""
    p = pathlib.Path(path)
    return p if p.is_absolute() else ROOT / p
