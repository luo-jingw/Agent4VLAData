"""Merge the arc-resampled picking and placement sets into one multi-task LeRobot root.

Picking and placement are trained together so one set of weights serves both, selected by the prompt.
Merging is mostly bookkeeping -- episodes renumbered, the global `index` column made continuous,
task strings concatenated and each episode's `task_index` remapped -- but two things are not:

  * The norm stats must be recomputed over the union. The two collections put the arm in different
    places (over the bin, over the conveyor), so statistics fitted to either one alone would
    mis-scale the other.
  * Videos are hard-linked, never re-encoded. They were already given a keyframe every second frame
    when each set was built, and re-encoding again would cost another lossy generation for nothing.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python merge_datasets.py --sources data/pick-arc-v1 data/place-arc-v1 --dst data/pickplace-arc-v1
"""
from __future__ import annotations
import dataclasses, json, os, pathlib, shutil
import numpy as np, pandas as pd, tyro

@dataclasses.dataclass(frozen=True)
class Args:
    sources: tuple[pathlib.Path, ...]
    dst: pathlib.Path
    repo_id: str = "world-studio/agibot-g2-pickplace-arc-v1"

def main(args: Args) -> None:
    if args.dst.exists(): shutil.rmtree(args.dst)
    (args.dst / "meta").mkdir(parents=True); (args.dst / "data" / "chunk-000").mkdir(parents=True)

    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in args.sources]
    base = infos[0]
    video_keys = [k for k, v in base["features"].items() if v["dtype"] == "video"]
    for i in infos[1:]:
        if i["features"].keys() != base["features"].keys():
            raise RuntimeError("sources disagree on their feature set")

    tasks, eps, stats, prov, new, run, total = [], [], [], [], 0, 0, 0
    for src, info in zip(args.sources, infos, strict=True):
        offset = len(tasks)
        src_tasks = [json.loads(l) for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]
        tasks += [{"task_index": offset + t["task_index"], "task": t["task"]} for t in src_tasks]
        src_stats = {int(json.loads(l)["episode_index"]): json.loads(l)["stats"]
                     for l in (src / "meta" / "episodes_stats.jsonl").read_text().splitlines() if l.strip()}
        report = json.loads((src / "meta" / "arc_resample.json").read_text())

        for path in sorted((src / "data" / "chunk-000").glob("*.parquet")):
            old = int(path.stem.split("_")[1])
            frame = pd.read_parquet(path)
            n = len(frame)
            ti = offset + int(frame["task_index"].iloc[0])
            frame["episode_index"] = np.full(n, new, dtype=np.int64)
            frame["index"] = np.arange(run, run + n, dtype=np.int64)
            frame["task_index"] = np.full(n, ti, dtype=np.int64)
            frame.to_parquet(args.dst / "data" / "chunk-000" / f"episode_{new:06d}.parquet", index=False)
            for key in video_keys:
                dst = args.dst / "videos" / "chunk-000" / key / f"episode_{new:06d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try: os.link(src / "videos" / "chunk-000" / key / f"episode_{old:06d}.mp4", dst)
                except OSError: shutil.copy2(src / "videos" / "chunk-000" / key / f"episode_{old:06d}.mp4", dst)
            eps.append({"episode_index": new, "tasks": [tasks[ti]["task"]], "length": n})
            st = dict(src_stats[old])
            for k, v in (("episode_index", np.full(n, new)), ("index", np.arange(run, run + n)),
                         ("task_index", np.full(n, ti))):
                f = np.asarray(v, dtype=np.float64)[:, None]
                st[k] = {"min": f.min(0).tolist(), "max": f.max(0).tolist(), "mean": f.mean(0).tolist(),
                         "std": f.std(0).tolist(), "count": [n]}
            stats.append({"episode_index": new, "stats": st})
            prov.append({"new_index": new, "source": str(src), "source_index": old,
                         "frames": n, "task_index": ti})
            new += 1; run += n; total += n
        prov.append({"source": str(src), "kept_episodes": report["kept_episodes"],
                     "config": report["config"], "task_kind": report.get("task_kind")})

    (args.dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (args.dst / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in stats))
    (args.dst / "meta" / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    out = dict(base)
    out.update({"repo_id": args.repo_id, "total_episodes": new, "total_frames": total,
                "total_tasks": len(tasks), "total_videos": new * len(video_keys),
                "splits": {"train": f"0:{new}"}})
    (args.dst / "meta" / "info.json").write_text(json.dumps(out, indent=2))
    (args.dst / "meta" / "arc_resample.json").write_text(json.dumps(
        {"merged_from": [str(s) for s in args.sources], "kept_episodes": new, "kept_frames": total,
         "config": json.loads((args.sources[0] / "meta" / "arc_resample.json").read_text())["config"],
         "episodes": prov}, indent=2))
    print(f"merged {new} episodes / {total} frames into {args.dst}")
    for t in tasks: print(f"  task {t['task_index']}: {t['task']}")
    counts = pd.Series([p["task_index"] for p in prov if "task_index" in p]).value_counts().sort_index()
    print("  episodes per task:", counts.to_dict())

if __name__ == "__main__":
    main(tyro.cli(Args))
