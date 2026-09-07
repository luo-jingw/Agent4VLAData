"""Turn the trained orbax checkpoints into the release tree that goes to the Hugging Face Hub.

openpi trains in JAX and saves orbax directories; the robot runtime loads one GGUF file; anything
that wants to poke at the weights in Python wants safetensors. Nothing converts orbax straight to
GGUF, so each step passes through PyTorch on the way:

    <step>/params      orbax, JAX          what training left behind  ->  jax/
    torch/             model.safetensors   intermediate and shipped   ->  torch/
    gguf/<variant>/    orion05.gguf        what the runtime loads     ->  gguf/<variant>/

`train_state/` is never copied: it is the optimizer state, useful only for resuming this exact run.

The GGUF exporter reports success having written a model beside a tokenizer it only warned about,
and the converter warns rather than fails when it cannot find the norm stats to copy -- a model with
no statistics is discovered on the robot as actions of the wrong scale, not as a missing file. So
every stage is checked here before the next one runs.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python export_hf.py --stage all
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Literal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "openpi_patch"))

import config_arc  # noqa: E402
import tyro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OPENPI = ROOT.parent.parent / "train" / "openpi"
GGML = ROOT.parent.parent / "train" / "pi05-ggml"

# Trained step -> the name it is published under.
STEPS: dict[str, int] = {"10k": 10000, "20k": 20000, "30k": 29999}

Stage = Literal["jax", "torch", "gguf", "all"]


@dataclasses.dataclass(frozen=True)
class QuantVariant:
    """One GGUF build. `flags` are passed to the exporter verbatim.

    The embedding table stays fp16 in every variant: the exporter calls quantizing it conservative
    and it is what the prompt is looked up in, so the two task strings would pay for it first. The
    exporter already skips the precision-sensitive layers inside whatever it is told to quantize
    (norms, the flow timestep embedding, and the action expert's input and output projections).
    """

    name: str
    flags: tuple[str, ...]


VARIANTS: tuple[QuantVariant, ...] = (
    QuantVariant("fp16", ()),
    QuantVariant("q8_0", ("--quant_llm", "q8", "--quant_vision", "q8", "--quant_action", "q8")),
    QuantVariant("q4_0", ("--quant_llm", "q4", "--quant_vision", "q4", "--quant_action", "q4")),
    QuantVariant("q4_K", ("--quant_llm", "q4k", "--quant_vision", "q4k", "--quant_action", "q4k")),
)

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
NORM_STATS_NAME = "norm_stats.json"
TOKENIZER_NAME = "tokenizer.model"
GGUF_NAME = "orion05.gguf"


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoints: pathlib.Path = ROOT / "checkpoints" / "g2_arc_v1" / "run1"
    out: pathlib.Path = ROOT / "hf_export"
    # From google/paligemma-3b-pt-224. Not a training artefact -- the runtime encodes the prompt
    # with it, so it ships beside every GGUF.
    tokenizer: pathlib.Path = pathlib.Path(
        "/home/user1/.cache/openpi/big_vision/paligemma_tokenizer.model"
    )
    readme: pathlib.Path | None = None
    stage: Stage = "all"
    only: str | None = None  # restrict to one published name, e.g. "30k"
    precision: str = "bfloat16"  # the converter's output dtype; the GGUF exporter casts to fp16


def _env() -> dict[str, str]:
    """CPU-only convert. q4_K needs the ggml build on LD_LIBRARY_PATH."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    ggml_lib = str(GGML / "build" / "third_party" / "ggml" / "src")
    env["LD_LIBRARY_PATH"] = ggml_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _run(argv: list[str], cwd: pathlib.Path, label: str) -> None:
    started = time.monotonic()
    print(f"    $ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=_env())
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    print(f"    {label} ok in {time.monotonic() - started:.0f}s", flush=True)


def _check(path: pathlib.Path, what: str) -> pathlib.Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{what}: missing or empty -> {path}")
    return path


def find_norm_stats(tree: pathlib.Path) -> pathlib.Path:
    """openpi keys assets by config name and repo id, so the file sits at an unpredictable depth."""
    found = next(tree.rglob(NORM_STATS_NAME), None)
    if found is None:
        raise RuntimeError(f"no {NORM_STATS_NAME} anywhere under {tree}")
    return found


def stage_jax(step_dir: pathlib.Path, dest: pathlib.Path) -> None:
    """The orbax params and the statistics that go with them. No train state."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("params", "assets"):
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(step_dir / name, target)
    metadata = step_dir / "_CHECKPOINT_METADATA"
    if metadata.exists():
        shutil.copy2(metadata, dest / "_CHECKPOINT_METADATA")
    find_norm_stats(dest)
    print(f"    jax ok ({sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / 2**30:.1f} GiB)")


def stage_torch(step_dir: pathlib.Path, dest: pathlib.Path, args: Args) -> pathlib.Path:
    """Run openpi's JAX-to-PyTorch converter, then check what it actually produced."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _run(
        [
            str(OPENPI / ".venv" / "bin" / "python"),
            str(ROOT / "openpi_patch" / "convert_wrapper.py"),
            "--checkpoint-dir", str(step_dir),
            "--output-path", str(dest),
            "--precision", args.precision,
        ],  # fmt: skip
        cwd=OPENPI,
        label="jax -> torch",
    )
    _check(dest / WEIGHTS_NAME, "converted weights")
    _check(dest / CONFIG_NAME, "converted config")
    find_norm_stats(dest)
    shutil.copyfile(args.tokenizer, dest / TOKENIZER_NAME)
    return dest


def stage_gguf(torch_dir: pathlib.Path, dest: pathlib.Path, args: Args) -> None:
    """One GGUF per quantization variant, each with the two files the runtime needs beside it."""
    _check(torch_dir / WEIGHTS_NAME, "torch weights (run --stage torch first)")
    _check(torch_dir / TOKENIZER_NAME, "tokenizer beside the weights")
    norm_stats = find_norm_stats(torch_dir)

    for variant in VARIANTS:
        out = dest / variant.name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        _run(
            [
                str(OPENPI / ".venv" / "bin" / "python"),
                str(GGML / "quantization" / "export_pi05.py"),
                "-d", str(torch_dir), "-o", str(out), *variant.flags,
            ],  # fmt: skip
            cwd=GGML,
            label=f"gguf {variant.name}",
        )
        # The exporter writes the model whether or not the tokenizer was there, so the delivery set
        # is assembled and checked here rather than trusted.
        shutil.copyfile(args.tokenizer, out / TOKENIZER_NAME)
        shutil.copyfile(norm_stats, out / NORM_STATS_NAME)
        for name in (GGUF_NAME, NORM_STATS_NAME, TOKENIZER_NAME):
            _check(out / name, f"gguf {variant.name} delivery set")
        size = (out / GGUF_NAME).stat().st_size / 2**30
        print(f"    {variant.name}: {GGUF_NAME} {size:.2f} GiB")


def main(args: Args) -> None:
    _check(args.tokenizer, "paligemma tokenizer")
    names = [args.only] if args.only else list(STEPS)
    manifest: dict[str, dict] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.readme is not None:
        shutil.copyfile(args.readme, args.out / "README.md")

    for name in names:
        step = STEPS[name]
        step_dir = args.checkpoints / str(step)
        if not step_dir.is_dir():
            raise RuntimeError(f"no checkpoint at {step_dir}")
        release = args.out / name
        print(f"\n=== {name} (step {step}) ===", flush=True)

        if args.stage in ("jax", "all"):
            stage_jax(step_dir, release / "jax")
        if args.stage in ("torch", "all"):
            stage_torch(step_dir, release / "torch", args)
        if args.stage in ("gguf", "all"):
            stage_gguf(release / "torch", release / "gguf", args)

        manifest[name] = {
            "step": step,
            "sizes_gib": {
                part.name: round(
                    sum(f.stat().st_size for f in part.rglob("*") if f.is_file()) / 2**30, 2
                )
                for part in sorted(release.iterdir())
                if part.is_dir()
            },
        }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "export_manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**previous, **manifest}, indent=2))
    print(f"\n{json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
