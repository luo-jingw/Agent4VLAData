"""Show the resampling as video: the same episode, played from the resampled frame order.

The action labels are built by walking a progress coordinate instead of the clock, so each output
sample names a source frame. Playing those frames back at 30 Hz is exactly what the robot would see
if it executed the resampled chunk -- pauses stepped over, grasp events at recorded speed, everything
else about 1.17x faster. Side by side with the original, the difference is watchable rather than
tabulated.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python resample_video.py --dataset data/new/suction-bin-picking --episode 0
"""
from __future__ import annotations
import dataclasses, pathlib, subprocess
import progress_resample as ar, av, numpy as np, pandas as pd, tyro

ROOT = pathlib.Path(__file__).resolve().parent

@dataclasses.dataclass(frozen=True)
class Args:
    dataset: pathlib.Path = ROOT / "data" / "new" / "suction-bin-picking"
    episode: int = 0
    camera: str = "image"
    min_dwell_frames: int = 0     # 0 = use each event's measured seal time, no artificial floor
    approach_frames: int = 15  # event windows stay at ratio 1.0; only free motion is scaled
    span_frames: int = 200     # recorded frames one chunk of 50 points aims to cover
    event_points: int = 32
    suffix: str = ""
    out_dir: pathlib.Path = ROOT / "out" / "videos"
    fps: int = 30

def write(frames_wanted: list[int], src: pathlib.Path, dst: pathlib.Path, fps: int) -> None:
    """Emit the requested source frames, in order, repeating any that are asked for twice.

    Decoded with PyAV and piped raw to ffmpeg: the frames are reordered and duplicated, so they carry
    timestamps the encoder cannot be handed directly.
    """
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        need = set(frames_wanted)
        cache = {}
        for i, frame in enumerate(inp.decode(stream)):
            if i in need:
                cache[i] = frame.to_ndarray(format="rgb24")
            if len(cache) == len(need):
                break
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)],
        stdin=subprocess.PIPE,
    )
    nearest = sorted(cache)
    for idx in frames_wanted:
        frame = cache.get(idx)
        if frame is None:
            frame = cache[min(nearest, key=lambda k: abs(k - idx))]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {dst}")


def main(args: Args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(args.dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet",
                            columns=["state", "actions"])
    states = np.stack(table["state"].to_numpy()).astype(np.float64)
    actions = np.stack(table["actions"].to_numpy()).astype(np.float64)
    cfg = ar.ResampleConfig(
        action_horizon=50,
        span_frames=args.span_frames,
        progress=ar.ProgressConfig(
            approach_frames=args.approach_frames,
            event_points=args.event_points,
            seal=ar.SealConfig(min_dwell_frames=args.min_dwell_frames),
        ),
    )
    resampled, curve = ar.resample(states, actions, cfg)
    picked = np.clip(np.rint(resampled.t).astype(int), 0, len(states) - 1).tolist()

    src = args.dataset / "videos" / "chunk-000" / args.camera / f"episode_{args.episode:06d}.mp4"
    tag = f"{args.dataset.name}_ep{args.episode:03d}{args.suffix}"
    original, resamp = args.out_dir / f"{tag}_original.mp4", args.out_dir / f"{tag}_resampled.mp4"
    write(list(range(len(states))), src, original, args.fps)
    write(picked, src, resamp, args.fps)

    label = ("[0:v]drawtext=text='ORIGINAL 30 Hz  %.1fs':fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:x=10:y=10[a];"
             "[1:v]drawtext=text='ARC-RESAMPLED  %.1fs':fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:x=10:y=10[b];"
             "[a][b]hstack=inputs=2") % (len(states) / args.fps, len(picked) / args.fps)
    side = args.out_dir / f"{tag}_sidebyside.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(original), "-i", str(resamp),
                    "-filter_complex", label, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(side)],
                   check=True)
    unique = len(set(picked))
    print(f"{tag}: {len(states)} frames ({len(states)/args.fps:.1f}s) -> {len(picked)} ({len(picked)/args.fps:.1f}s), "
          f"speedup {len(states)/len(picked):.3f}x, {unique} distinct source frames used "
          f"({unique/len(states)*100:.0f}% of the episode), windows {len(curve.windows)}")
    print(f"  {side}")

if __name__ == "__main__":
    main(tyro.cli(Args))
