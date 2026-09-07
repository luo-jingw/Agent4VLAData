"""Per-episode, per-dimension variance attribution, for manual dataset review.

The question a reviewer asks is "is any dimension driven by one episode", and that
question lives on an (episode x dimension) matrix. A chart that plots one value per
dimension cannot answer it: it has already summed over episodes.

Two views of that matrix, side by side. The radar draws one translucent polygon per
episode over the dimension axes, so an episode that dominates a dimension leaves a
spike out of an otherwise tight bundle; the bold envelope is the per-dimension
maximum. The matrix names which episode it was.

The cell value is the share of a dimension's variance attributable to one episode:

    share(e, d) = 1 - Var_d(all frames except episode e) / Var_d(all frames)

A fair share is 1/E (0.007 for 150 episodes). A cell at 1.0 means removing that
episode removes the dimension's entire spread.

Leave-one-out variance is closed form and O(N): accumulate (n, S1, S2) per episode,
then subtract. The accumulation is done on centred values because dimensions exist
whose values are around 1.73 while their variance is 1e-13, and the textbook
`sum(x^2) - sum(x)^2/n` cancels catastrophically there.

A low ratio says the episode dominates that dimension's variance. It does NOT say
the episode is anomalous -- it may be the only place the dimension carries signal.
The strip under each heatmap carries `span_ratio` so the reviewer can tell a
dimension whose variance is physically meaningful from one that has none, and the
`moved` count says whether other episodes would still cover the dimension if this
one were dropped. Nothing here is compared against a threshold.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm

from joint_limits import LIMIT_RANGE

NON_JOINT_MARKERS = ("suction", "vacuum")
SHARE_CMAP = "magma_r"
SPAN_CMAP = "viridis"


@dataclass(frozen=True)
class Attribution:
    """One feature's variance attribution over episodes and dimensions."""

    repo_id: str
    feature: str
    frames: int
    names: tuple[str, ...]
    episodes: np.ndarray            # (E,)
    share: np.ndarray               # (E, D) in [0, 1]; NaN where the dim is constant
    span_ratio: np.ndarray          # (D,) NaN where the dim has no joint limit
    moved: np.ndarray               # (D,) episodes in which the dim moves
    total_variance: np.ndarray      # (D,)

    def select(self, dims: str) -> Attribution:
        if dims == "all":
            return self
        want = dims == "joints"
        keep = [
            i
            for i, name in enumerate(self.names)
            if (not any(m in name for m in NON_JOINT_MARKERS)) == want
        ]
        return Attribution(
            self.repo_id,
            self.feature,
            self.frames,
            tuple(self.names[i] for i in keep),
            self.episodes,
            self.share[:, keep],
            self.span_ratio[keep],
            self.moved[keep],
            self.total_variance[keep],
        )


def _read(source: Path, feature: str) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    columns = [feature, "episode_index"]
    if source.is_dir():
        info = json.loads((source / "meta" / "info.json").read_text())
        frames = [
            pl.read_parquet(path, columns=columns)
            for path in sorted(glob.glob(str(source / "data" / "chunk-*" / "*.parquet")))
        ]
    else:
        with tarfile.open(source) as archive:
            info = json.loads(archive.extractfile("meta/info.json").read())
            members = sorted(
                (m for m in archive.getmembers() if m.name.endswith(".parquet")),
                key=lambda m: m.name,
            )
            frames = [
                pl.read_parquet(io.BytesIO(archive.extractfile(m).read()), columns=columns)
                for m in members
            ]
    if not frames:
        raise FileNotFoundError(f"no parquet data in {source}")
    values = np.concatenate(
        [np.stack(frame[feature].to_numpy()) for frame in frames]
    ).astype(np.float64)
    episodes = np.concatenate(
        [np.full(len(frame), int(frame["episode_index"][0])) for frame in frames]
    )
    return values, episodes, list(info["features"][feature]["names"]), str(info["repo_id"])


def attribute(source: Path, feature: str) -> Attribution:
    values, episode_of_frame, names, repo_id = _read(source, feature)
    ids = np.unique(episode_of_frame)

    # Centre before accumulating second moments: some dimensions sit at 1.73 with a
    # variance of 1e-13, where sum(x^2) - sum(x)^2/n loses every significant digit.
    centred = values - values.mean(axis=0)
    total = len(centred)
    sum1, sum2 = centred.sum(axis=0), (centred**2).sum(axis=0)

    counts = np.array([(episode_of_frame == e).sum() for e in ids])
    sum1_e = np.stack([centred[episode_of_frame == e].sum(axis=0) for e in ids])
    sum2_e = np.stack([(centred[episode_of_frame == e] ** 2).sum(axis=0) for e in ids])

    remaining = (total - counts)[:, None]
    variance_loo = np.maximum((sum2 - sum2_e) / remaining - ((sum1 - sum1_e) / remaining) ** 2, 0.0)
    variance_all = np.maximum(sum2 / total - (sum1 / total) ** 2, 0.0)

    constant = variance_all <= 0.0
    share = 1.0 - variance_loo / np.where(constant, 1.0, variance_all)
    share[:, constant] = np.nan

    span_ratio = np.array(
        [
            (values[:, i].max() - values[:, i].min()) / LIMIT_RANGE[name]
            if name in LIMIT_RANGE
            else np.nan
            for i, name in enumerate(names)
        ]
    )
    moved = np.array(
        [
            sum(
                1
                for e in ids
                if values[episode_of_frame == e, i].max() > values[episode_of_frame == e, i].min()
            )
            for i in range(values.shape[1])
        ]
    )
    return Attribution(
        repo_id, feature, total, tuple(names), ids, share, span_ratio, moved, variance_all
    )


_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("left_arm_joint_", "L_arm"),
    ("right_arm_joint_", "R_arm"),
    ("left_vacuum_ch", "L_vac"),
    ("right_vacuum_ch", "R_vac"),
    ("left_suction_enabled", "L_suction"),
    ("right_suction_enabled", "R_suction"),
)


def short_name(name: str) -> str:
    for prefix, short in _ABBREVIATIONS:
        if name.startswith(prefix):
            return name.replace(prefix, short, 1)
    return name


@dataclass(frozen=True)
class Candidate:
    """An (episode, dimension) pair worth a human look, with what removal would cost."""

    episode: int
    dimension: str
    share: float
    span_ratio: float
    moved: int
    total_episodes: int

    @property
    def consequence(self) -> str:
        """What dropping this episode does to the dimension. Not a recommendation."""
        return "removal leaves the dim constant" if self.moved <= 1 else (
            f"removal leaves {self.moved - 1}/{self.total_episodes} episodes still moving it"
        )


def candidates(data: Attribution, share_floor: float = 0.5,
               span_floor: float = 1e-3) -> list[Candidate]:
    """Cells where one episode dominates a dimension whose spread is physically real.

    `span_floor` is deliberately the same constant the span floor uses: a dimension
    below it is one the floor already neutralises, so a high share there describes
    noise, not data. `share_floor` is what "dominates" means; both are reported
    alongside the numbers so a reviewer can disagree with either.
    """
    filled = np.nan_to_num(data.share, nan=0.0)
    found = []
    for column, name in enumerate(data.names):
        ratio = data.span_ratio[column]
        if np.isnan(ratio) or ratio <= span_floor:
            continue
        for row in np.flatnonzero(filled[:, column] >= share_floor):
            found.append(
                Candidate(int(data.episodes[row]), name, float(filled[row, column]),
                          float(ratio), int(data.moved[column]), len(data.episodes))
            )
    return sorted(found, key=lambda c: c.share, reverse=True)


def _draw_radar(axis, data: Attribution, share_max: float, highlight: int) -> None:
    """One translucent polygon per episode over the dimension axes."""
    count = len(data.names)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    wrapped = np.concatenate([angles, angles[:1]])
    filled = np.nan_to_num(data.share, nan=0.0)
    fair = 1.0 / len(data.episodes)

    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_ylim(0.0, max(share_max, filled.max()) * 1.05)
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.spines["polar"].set_color("#cccccc")

    # Filled and stacked: where the polygons pile up the alpha accumulates, so the
    # bulk reads as a dark core at the centre and a lone spike stays a thin wedge.
    for row in range(len(data.episodes)):
        ring = np.concatenate([filled[row], filled[row, :1]])
        axis.fill(wrapped, ring, color="#6f6f6f", alpha=0.05, linewidth=0)
        axis.plot(wrapped, ring, color="#6f6f6f", alpha=0.16, linewidth=0.6)

    axis.plot(wrapped, np.full(count + 1, fair), color="#3aa0ff",
              linewidth=1.2, linestyle="--", zorder=4)

    envelope = filled.max(axis=0)
    axis.plot(wrapped, np.concatenate([envelope, envelope[:1]]),
              color="#111111", linewidth=1.8, zorder=5)

    ranked = np.argsort(filled.max(axis=1))[::-1][:highlight]
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    flagged = {c.episode: c for c in candidates(data)}
    for rank, row in enumerate(ranked):
        column = int(filled[row].argmax())
        ring = np.concatenate([filled[row], filled[row, :1]])
        colour = palette[rank % 10]
        axis.fill(wrapped, ring, color=colour, alpha=0.30, linewidth=0, zorder=6)
        axis.plot(wrapped, ring, color=colour, linewidth=1.9, zorder=7)
        mark = flagged.get(int(data.episodes[row]))
        note = "" if mark is None else f"   [{mark.consequence}]"
        axis.plot([], [], color=colour, linewidth=1.9,
                  label=f"ep{data.episodes[row]}  ·  {short_name(data.names[column])}"
                        f"  ·  {filled[row, column]:.3f}{note}")

    # The span/limit ratio rides on the axis label: without it a spike on a dead
    # dimension (span/limit ~ 1e-5) looks exactly like one on a live dimension.
    axis.set_xticks(angles)
    axis.set_xticklabels(
        [
            f"{short_name(n)}\n{'—' if np.isnan(r) else f'{r:.0e}'}"
            for n, r in zip(data.names, data.span_ratio)
        ],
        fontsize=7.5,
    )
    axis.tick_params(labelsize=7.5, pad=2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8.5,
                frameon=False, ncol=2, title="highest share", title_fontsize=8.5)
    axis.set_title(
        f"{data.feature}   ·   one polygon per episode   ·   axis label carries span/limit"
        f"\nblack = per-dimension max   ·   dashed = fair share 1/E = {fair:.4f}",
        fontsize=10.5, pad=26,
    )


def _draw_feature(figure, spec, data: Attribution, share_max: float, label_x: bool):
    """One feature: per-dimension bar on top, the matrix, per-episode bar on the right."""
    grid = spec.subgridspec(
        2, 3, height_ratios=[0.17, 1.0], width_ratios=[1.0, 0.10, 0.03],
        hspace=0.05, wspace=0.04,
    )
    count = len(data.names)
    filled = np.nan_to_num(data.share, nan=0.0)
    per_dim = filled.max(axis=0)
    per_episode = filled.max(axis=1)
    fair = 1.0 / len(data.episodes)

    top = figure.add_subplot(grid[0, 0])
    top.bar(np.arange(count), per_dim, color="#3b3b3b", width=0.78)
    top.axhline(fair, color="#3aa0ff", linewidth=1.0, linestyle="--")
    top.set_xlim(-0.5, count - 0.5)
    top.set_ylim(0.0, 1.05)
    top.set_xticks([])
    top.set_yticks([0.0, 0.5, 1.0])
    top.tick_params(labelsize=7)
    top.set_ylabel("max share", fontsize=8)
    for spine in ("top", "right"):
        top.spines[spine].set_visible(False)
    top.set_title(
        f"{data.feature}   ·   {len(data.episodes)} episodes × {count} dims"
        f"   ·   dashed = fair share 1/E = {fair:.4f}",
        fontsize=12, pad=10,
    )

    heat = figure.add_subplot(grid[1, 0])
    image = heat.imshow(
        data.share, aspect="auto", origin="lower", cmap=SHARE_CMAP,
        vmin=0.0, vmax=share_max, interpolation="nearest",
        extent=(-0.5, count - 0.5, data.episodes.min() - 0.5, data.episodes.max() + 0.5),
    )
    heat.set_ylabel("episode index", fontsize=9)
    heat.tick_params(labelsize=8)
    heat.set_xticks(np.arange(count))
    if label_x:
        # Three facts per dimension, so the matrix needs no second colour strip:
        # name, in how many episodes it moves, and its span against the joint limit.
        heat.set_xticklabels(
            [
                f"{short_name(n)}   {m}/{len(data.episodes)}   "
                + ("span/lim —" if np.isnan(r) else f"span/lim {r:.1e}")
                for n, m, r in zip(data.names, data.moved, data.span_ratio)
            ],
            rotation=90, fontsize=8,
        )
    else:
        heat.set_xticklabels([])

    order = np.argsort(filled, axis=None)[::-1][:4]
    for rank, index in enumerate(order):
        row, column = np.unravel_index(index, filled.shape)
        span = len(data.episodes)
        heat.annotate(
            f"ep{data.episodes[row]} · {short_name(data.names[column])} · {filled[row, column]:.3f}",
            xy=(column, data.episodes[row]),
            xytext=(count * 0.62, data.episodes.min() + span * (0.94 - 0.075 * rank)),
            fontsize=8.5, color="#c81d3a", va="center",
            arrowprops={"arrowstyle": "-", "color": "#c81d3a", "linewidth": 0.8},
        )

    side = figure.add_subplot(grid[1, 1], sharey=heat)
    side.barh(data.episodes, per_episode, color="#3b3b3b", height=1.0)
    side.axvline(fair, color="#3aa0ff", linewidth=1.0, linestyle="--")
    side.set_xlim(0.0, 1.05)
    side.set_xticks([0.0, 1.0])
    side.tick_params(labelsize=7, labelleft=False)
    side.set_xlabel("max share\nper episode", fontsize=8)
    for spine in ("top", "right"):
        side.spines[spine].set_visible(False)

    bar = figure.colorbar(image, cax=figure.add_subplot(grid[1, 2]))
    bar.set_label("share of the dimension's variance", fontsize=8.5)
    bar.ax.tick_params(labelsize=7.5)


def render(per_feature: dict[str, Attribution], destination: Path,
           share_max: float, highlight: int) -> None:
    features = [f for f in ("state", "actions") if f in per_feature]
    count = len(per_feature[features[0]].names)
    figure = plt.figure(figsize=(12.0 + 0.60 * count, 9.4 * len(features)))
    outer = figure.add_gridspec(len(features), 2, width_ratios=[1.0, 1.2],
                                hspace=0.30, wspace=0.20)
    for row, feature in enumerate(features):
        data = per_feature[feature]
        _draw_radar(figure.add_subplot(outer[row, 0], projection="polar"),
                    data, share_max, highlight)
        _draw_feature(figure, outer[row, 1], data, share_max,
                      label_x=row == len(features) - 1)
    any_data = per_feature[features[0]]
    subtitle = "leave-one-out variance attribution"
    if len(features) == 2:
        a, b = (np.nan_to_num(per_feature[f].share, nan=0.0) for f in features)
        if a.shape == b.shape:
            subtitle += f"   ·   state vs actions, max |Δshare| = {np.abs(a - b).max():.1e}"
    figure.suptitle(
        f"{any_data.repo_id}   ·   {any_data.frames} frames   ·   {subtitle}",
        fontsize=14, y=0.995,
    )
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeRobot directory or .tar.gz of one")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dims", default="joints", choices=("joints", "end_effector", "all"))
    parser.add_argument(
        "--share-max",
        type=float,
        default=1.0,
        help="upper end of the colour scale; keep equal across datasets being compared",
    )
    parser.add_argument(
        "--highlight", type=int, default=4,
        help="how many episodes to draw in colour and name in the radar legend",
    )
    arguments = parser.parse_args()

    selector = {"joints": "joints", "end_effector": "ee", "all": "all"}
    per_feature: dict[str, Attribution] = {}
    for feature in ("state", "actions"):
        full = attribute(arguments.source, feature)
        chosen = full.select(selector[arguments.dims])
        if chosen.names:
            per_feature[feature] = chosen
        flat = np.nan_to_num(full.share, nan=0.0)
        order = np.argsort(flat.max(axis=0))[::-1]
        print(f"{full.repo_id}  {feature}  frames={full.frames}  episodes={len(full.episodes)}")
        for column in order[:8]:
            row = int(flat[:, column].argmax())
            constant = "  constant" if np.isnan(full.share[:, column]).all() else ""
            print(
                f"  {full.names[column]:<22} max share={flat[row, column]:.4f}"
                f"  ep={full.episodes[row]:<5} span/limit={full.span_ratio[column]:.2e}"
                f"  moved={full.moved[column]}/{len(full.episodes)}{constant}"
            )

    for feature, data in per_feature.items():
        found = candidates(data)
        print(f"-- {feature}: {len(found)} candidate(s) for review "
              f"(share >= 0.5 and span/limit > 1e-3)")
        for item in found:
            print(f"     ep{item.episode:<5} {item.dimension:<20} share={item.share:.3f}"
                  f"  span/limit={item.span_ratio:.2e}"
                  f"  moved={item.moved}/{item.total_episodes}  -> {item.consequence}")

    render(per_feature, arguments.output, arguments.share_max, arguments.highlight)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
