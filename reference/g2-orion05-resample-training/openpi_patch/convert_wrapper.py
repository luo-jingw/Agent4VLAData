"""Run openpi's JAX-to-PyTorch converter against a config that is not in openpi's own registry.

`examples/convert_jax_model_to_pytorch.py` resolves the model by name through
`openpi.training.config.get_config`, and it has to be the config that was trained with -- the
converter rebuilds the architecture from it rather than reading it out of the checkpoint. `g2_arc_v1`
lives outside the vendored tree, so it is registered here first and the converter's `main` is then
called directly, skipping its CLI.

  ../../train/openpi/.venv/bin/python openpi_patch/convert_wrapper.py \
      --checkpoint-dir <step dir> --output-path <dest> --precision bfloat16
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OPENPI = ROOT.parent.parent / "train" / "openpi"

import config_arc  # noqa: E402
import tyro  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: pathlib.Path
    output_path: pathlib.Path
    precision: str = "bfloat16"


def _load_converter() -> types.ModuleType:
    path = OPENPI / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_convert_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args: Args) -> None:
    config = config_arc.register(ROOT)
    converter = _load_converter()
    converter.main(
        checkpoint_dir=str(args.checkpoint_dir),
        config_name=config.name,
        output_path=str(args.output_path),
        precision=args.precision,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
