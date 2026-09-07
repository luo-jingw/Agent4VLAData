"""Build an arc-resampled training set from one of the single-task G2 collections.

Same three jobs as build_dataset.py -- clean, keep the prompt, add the arc-length chunk column -- but
these collections have one task string and one arm, so the two-prompt machinery of the task_2 builder
does not apply and its cleaning rules do not transfer:

  picking    150 episodes, each exactly one grasp held to the last frame, no releases at all.
  placement  137 episodes, each starting with the part already held and ending in a release. Here a
             release is the task, so task_2's rule -- "a released grip is a failed grasp, drop the
             episode" -- would delete 136 of 137. What marks a failure here is the opposite: an
             episode that never releases.

Kept from task_2: rules are read off the data rather than a hand-written list, and the duration upper
tail is trimmed within a group rather than globally.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python build_new_dataset.py \
      --src data/new/suction-bin-picking --dst data/pick-arc-v1 --task-kind pick
"""
from __future__ import annotations
import concurrent.futures, dataclasses, json, os, pathlib, shutil, subprocess, time
from typing import Literal
import progress_resample as ar, numpy as np, pandas as pd, tyro

TaskKind = Literal["pick", "place"]
VideoMode = Literal["link", "reencode"]

@dataclasses.dataclass(frozen=True)
class Args:
    src: pathlib.Path
    dst: pathlib.Path
    task_kind: TaskKind
    repo_id: str | None = None
    action_horizon: int = 50
    span_frames: int = 200     # recorded frames one chunk of `action_horizon` points aims to cover
    event_points: int = 32     # points given to one grasp event, spread evenly in time
    deadzone: float = 0.002
    approach_frames: int = 15
    min_dwell_frames: int = 0
    duration_quantile: float = 0.95
    key_weight: float = 3.0   # loss multiplier on chunk steps drawn from an event window
    metric: str = "curvature"  # spacing rule outside the event windows; see MetricConfig
    # Frames of the frozen opening to keep. Every recording starts with ~3 s of a parked arm before
    # the operator moves -- a fifth of the episode, and the same observation each time, since the
    # joints are constant and the cameras differ only by noise. The resampling already denies those
    # frames action points, but they stay in the table as rows, so a fifth of every batch is one
    # repeated sample. Keeping half a second leaves the policy plenty of evidence for how to start.
    trim_lead_frames: int | None = 15
    lead_still_eps: float = 1e-3  # rad per frame, on measured and commanded joints alike
    video_mode: VideoMode = "reencode"
    crf: int = 23             # x264 quality for the re-encode
    encode_jobs: int = 12     # videos re-encoded in parallel
    encode_threads: int = 4   # x264 threads per video

@dataclasses.dataclass(frozen=True)
class Audit:
    index: int
    frames: int
    duration_s: float
    grasps_held: int      # grips still holding at the last frame
    releases: int         # grips that were let go
    keep: bool = True
    reason: str = "keep"

def audit_one(actions: np.ndarray) -> tuple[int, int]:
    held = released = 0
    n = len(actions)
    for dim in ar.SUCTION_ACTION_DIMS:
        on = (actions[:, dim] > 0.5).astype(np.int8)
        edges = np.diff(on)
        starts = ([0] if on[0] else []) + (np.flatnonzero(edges > 0) + 1).tolist()
        ends = (np.flatnonzero(edges < 0) + 1).tolist()
        for s in starts:
            after = [e for e in ends if e > s]
            if after: released += 1
            else: held += 1
    return held, released

def decide(rows: list[Audit], args: Args) -> list[Audit]:
    out = []
    for r in rows:
        # What "finished the task" means is the collection's own definition, not a shared constant.
        ok = r.grasps_held >= 1 if args.task_kind == "pick" else r.releases >= 1
        out.append(r if ok else dataclasses.replace(r, keep=False, reason="incomplete"))
    kept = [r.duration_s for r in out if r.keep]
    limit = float(np.quantile(kept, args.duration_quantile)) if kept else float("inf")
    return [dataclasses.replace(r, keep=False, reason="slow")
            if r.keep and r.duration_s > limit else r for r in out]

def _stats(v: np.ndarray) -> dict:
    f = v.reshape(len(v), -1).astype(np.float64)
    return {"min": f.min(0).tolist(), "max": f.max(0).tolist(), "mean": f.mean(0).tolist(),
            "std": f.std(0).tolist(), "count": [len(f)]}

def _place(src, dst, mode, crf, threads, start=0):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    if mode == "link":
        try: os.link(src, dst); return
        except OSError: shutil.copy2(src, dst); return
    # -g 2 puts a keyframe every other frame. LeRobot reads frames in random order, and inheriting the
    # recording's ~190-frame GOP costs about 95 extra decodes per read: 2.08 s/it against 0.97.
    trim = ["-vf", f"select=gte(n\\,{start}),setpts=PTS-STARTPTS"] if start else []
    subprocess.run(["ffmpeg","-v","error","-y","-i",str(src),*trim,"-c:v","libx264","-crf",str(crf),
                    "-pix_fmt","yuv420p","-g","2","-keyint_min","2","-sc_threshold","0",
                    "-preset","fast","-threads",str(threads),str(dst)], check=True)
    def count(p):
        o = subprocess.run(["ffprobe","-v","error","-select_streams","v","-count_frames",
                            "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(p)],
                           capture_output=True, text=True, check=True)
        return int(o.stdout.strip())
    # The video and the parquet are indexed by the same frame number, so a miscount is silent
    # mislabelling rather than a crash: every observation would be paired with someone else's action.
    if count(dst) != count(src) - start:
        raise RuntimeError(f"{dst.name}: expected {count(src) - start} frames, got {count(dst)}")

def main(args: Args) -> None:
    info = json.loads((args.src / "meta" / "info.json").read_text())
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    src_stats = {int(json.loads(l)["episode_index"]): json.loads(l)["stats"]
                 for l in (args.src / "meta" / "episodes_stats.jsonl").read_text().splitlines() if l.strip()}
    tasks = [json.loads(l) for l in (args.src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]

    rows = []
    for p in sorted((args.src / "data" / "chunk-000").glob("*.parquet")):
        a = np.stack(pd.read_parquet(p, columns=["actions"])["actions"].to_numpy()).astype(np.float64)
        held, rel = audit_one(a)
        rows.append(Audit(int(p.stem.split("_")[1]), len(a), len(a)/ar.FPS, held, rel))
    rows = decide(rows, args)
    kept = [r for r in rows if r.keep]
    print(f"audited {len(rows)}, keeping {len(kept)}")
    for reason in ("incomplete", "slow"):
        d = [r.index for r in rows if r.reason == reason]
        print(f"  dropped by {reason:10s}: {len(d):3d} {d}")

    cfg = ar.ResampleConfig(action_horizon=args.action_horizon, span_frames=args.span_frames,
        metric=ar.MetricConfig(metric=args.metric),
        progress=ar.ProgressConfig(deadzone=args.deadzone, approach_frames=args.approach_frames,
            event_points=args.event_points,
            seal=ar.SealConfig(min_dwell_frames=args.min_dwell_frames)))

    if args.trim_lead_frames is not None and args.video_mode == "link":
        raise SystemExit("--trim-lead-frames needs --video-mode reencode: a hard link cannot drop frames")

    if args.dst.exists(): shutil.rmtree(args.dst)
    (args.dst / "meta").mkdir(parents=True); (args.dst / "data" / "chunk-000").mkdir(parents=True)
    eps, stats_meta, prov, jobs, run, total, trims = [], [], [], [], 0, 0, []
    for new, row in enumerate(kept):
        f = pd.read_parquet(args.src / "data" / "chunk-000" / f"episode_{row.index:06d}.parquet")
        S = np.stack(f["state"].to_numpy()).astype(np.float64)
        A = np.stack(f["actions"].to_numpy()).astype(np.float64)
        # On the whole recording, always. The trim below drops rows, and dropping them here instead
        # would divide the same progress among fewer free frames and shift `step` -- re-timing the
        # trajectory, when the intent is only to stop showing the model the same frame a hundred times.
        curve = ar.progress_curve(S, A, cfg)
        chunks = ar.all_chunks(A, curve, args.action_horizon).astype(np.float32)
        weights = ar.all_chunk_weights(curve, args.action_horizon, args.key_weight).astype(np.float32)
        source_frames = len(f); ti = int(f["task_index"].iloc[0])

        start = 0
        if args.trim_lead_frames is not None:
            frozen = ar.frozen_prefix(S, A, args.lead_still_eps)
            start = max(frozen - args.trim_lead_frames, 0)
        trims.append(start)
        S, A, chunks, weights = S[start:], A[start:], chunks[start:], weights[start:]
        n = len(S)

        pd.DataFrame({
            "state": [x for x in S.astype(np.float32)],
            "actions": [x for x in A.astype(np.float32)],
            "action_chunk": [c.reshape(-1) for c in chunks],
            "action_loss_weight": [w for w in weights],
            "timestamp": (np.arange(n)/ar.FPS).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, new, dtype=np.int64),
            "index": np.arange(run, run+n, dtype=np.int64),
            "task_index": np.full(n, ti, dtype=np.int64),
        }).to_parquet(args.dst / "data" / "chunk-000" / f"episode_{new:06d}.parquet", index=False)
        for k in video_keys:
            jobs.append((args.src/"videos"/"chunk-000"/k/f"episode_{row.index:06d}.mp4",
                         args.dst/"videos"/"chunk-000"/k/f"episode_{new:06d}.mp4", start))
        eps.append({"episode_index": new, "tasks": [tasks[ti]["task"]], "length": n})
        st = dict(src_stats[row.index]); st["state"] = _stats(S); st["actions"] = _stats(A)
        st["action_chunk"] = _stats(chunks.reshape(n, -1))
        st["action_loss_weight"] = _stats(weights)
        for k, v in (("timestamp", np.arange(n)/ar.FPS), ("frame_index", np.arange(n)),
                     ("episode_index", np.full(n, new)), ("index", np.arange(run, run+n)),
                     ("task_index", np.full(n, ti))):
            st[k] = _stats(np.asarray(v)[:, None])
        stats_meta.append({"episode_index": new, "stats": st})
        # compression_ratio stays on the recording's own length: it describes the resampling, which
        # the trim does not touch, and has to stay comparable with builds that trimmed differently.
        prov.append({"new_index": new, "source_index": row.index, "frames": n,
                     "source_frames": source_frames, "trimmed": start,
                     "samples_equivalent": curve.n_samples,
                     "compression_ratio": round(source_frames/max(curve.n_samples,1), 3),
                     "grasps_held": row.grasps_held, "releases": row.releases, "task_index": ti,
                     "step": round(curve.step, 6),
                     "event_frames": int((curve.approach_mask | curve.dwell_mask).sum()),
                     "windows": [{"kind": w.kind, "onset": w.onset, "measured": w.measured_frames,
                                  "used": w.frames} for w in curve.windows]})
        run += n; total += n

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.encode_jobs) as pool:
        for fut in concurrent.futures.as_completed(
            [pool.submit(_place, s, d, args.video_mode, args.crf, args.encode_threads, k)
             for s, d, k in jobs]):
            fut.result()
    print(f"{len(jobs)} videos placed by {args.video_mode} in {time.monotonic()-t0:.0f}s")

    (args.dst/"meta"/"episodes.jsonl").write_text("".join(json.dumps(e)+"\n" for e in eps))
    (args.dst/"meta"/"episodes_stats.jsonl").write_text("".join(json.dumps(e)+"\n" for e in stats_meta))
    (args.dst/"meta"/"tasks.jsonl").write_text("".join(json.dumps(t)+"\n" for t in tasks))
    out = dict(info)
    out.update({"repo_id": args.repo_id or f"world-studio/{args.dst.name}", "total_episodes": len(kept),
                "total_frames": total, "total_videos": len(kept)*len(video_keys),
                "splits": {"train": f"0:{len(kept)}"}, "conversion_template": "agibot_g2_arc_v1"})
    out["features"] = dict(info["features"])
    out["features"]["action_chunk"] = {"dtype": "float32", "names": None,
        "shape": [args.action_horizon * len(info["features"]["actions"]["names"])]}
    out["features"]["action_loss_weight"] = {"dtype": "float32", "names": None,
        "shape": [args.action_horizon]}
    (args.dst/"meta"/"info.json").write_text(json.dumps(out, indent=2))
    (args.dst/"meta"/"arc_resample.json").write_text(json.dumps({
        "source": str(args.src), "task_kind": args.task_kind,
        "config": dataclasses.asdict(args) | {"src": str(args.src), "dst": str(args.dst)},
        "kept_episodes": len(kept), "kept_frames": total,
        "dropped": {r: [x.index for x in rows if x.reason == r] for r in ("incomplete", "slow")},
        "audit": [dataclasses.asdict(r) for r in rows], "episodes": prov}, indent=2))
    sp = [e["compression_ratio"] for e in prov]
    print(f"  key weight {args.key_weight} on event-window chunk steps")
    if args.trim_lead_frames is not None:
        source_total = sum(e["source_frames"] for e in prov)
        print(f"  trimmed the frozen opening to {args.trim_lead_frames} frames: dropped "
              f"{source_total - total} of {source_total} rows ({(source_total - total) / source_total:.1%}), "
              f"p50 {int(np.median(trims))} per episode; labels unchanged")
    print(f"wrote {len(kept)} episodes / {total} frames to {args.dst}")
    print(f"  compression (source frames per action point) min {min(sp):.2f} median {np.median(sp):.2f} max {max(sp):.2f}")

if __name__ == "__main__":
    main(tyro.cli(Args))
