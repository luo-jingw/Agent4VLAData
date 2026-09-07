"""Run openpi training on the arc-resampled G2 dataset without modifying the vendored openpi tree.

openpi resolves a config by name out of a module-level registry, so registering ours at import time
is enough: `train.py`'s `main` takes the TrainConfig object directly. The vendored tree under
`train/openpi/` stays byte-identical -- nothing is pasted into `config.py`.

  cd tmp/tmp_train_G2
  WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ../../train/openpi/.venv/bin/python openpi_patch/register.py --exp-name run1 --overwrite

`--dry-run` stops after building the data loader and printing one batch's shapes, which is the
cheap way to confirm the chunk column survives the LeRobot -> openpi path.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent  # tmp/tmp_train_G2
OPENPI = ROOT.parent.parent / "train" / "openpi"
sys.path.insert(0, str(HERE))

import config_arc  # noqa: E402
import tyro  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Args:
    exp_name: str = "run1"
    num_train_steps: int = 30_000
    fsdp_devices: int = 2  # shard the model across the visible devices, as the G1 run did
    save_interval: int = 10_000
    overwrite: bool = False
    resume: bool = False
    dry_run: bool = False
    # Where LeRobot looks for the dataset. The build script writes into tmp/tmp_train_G2/data, so a
    # link is needed unless one already exists.
    dataset_dir: pathlib.Path = ROOT / "data" / "pickplace-arc-v4"


def _load_train_module() -> types.ModuleType:
    """Import train/openpi/scripts/train.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("openpi_train_script", OPENPI / "scripts" / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OPENPI / 'scripts' / 'train.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dataset_link(dataset_dir: pathlib.Path) -> pathlib.Path:
    """Point ~/.cache/huggingface/lerobot/<repo_id> at the built dataset."""
    if not (dataset_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{dataset_dir} is not a LeRobot root -- run build_dataset.py first")
    target = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / config_arc.REPO_ID
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == dataset_dir.resolve():
            return target
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"{target} exists and is not a link to {dataset_dir}")
    target.symlink_to(dataset_dir)
    return target


def shim_dataset_weights() -> bool:
    """Give DataConfig the `dataset_weights` field the vendored data loader already reads.

    The multi-dataset mixing work is mid-flight in the delivery tree: `data_loader.py` is committed
    reading `data_config.dataset_weights`, while the working copy of `config.py` no longer declares
    it, so every new training process dies on an AttributeError before the first step. This run mixes
    nothing -- one merged repo, weights irrelevant -- so a class-level default is enough to get past
    it, and `if weights:` then takes the same path it always did.

    A class attribute rather than an edit, because the vendored tree is someone else's work in
    progress and must stay untouched; this is the same run-time injection register.py already does
    for the config registry.
    """
    from openpi.training import config as _config

    if hasattr(_config.DataConfig, "dataset_weights"):
        return False
    _config.DataConfig.dataset_weights = ()
    return True


def main(args: Args) -> None:
    if shim_dataset_weights():
        print("shim: DataConfig.dataset_weights = () -- the vendored config.py is mid-edit and "
              "no longer declares it, while the committed data_loader.py reads it")

    link = ensure_dataset_link(args.dataset_dir)
    print(f"dataset: {link} -> {args.dataset_dir}")

    config = config_arc.register(
        ROOT,
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        fsdp_devices=args.fsdp_devices,
        save_interval=args.save_interval,
    )
    config = dataclasses.replace(config, overwrite=args.overwrite, resume=args.resume)
    print(
        f"config: {config.name}/{config.exp_name}  horizon {config.model.action_horizon}  "
        f"batch {config.batch_size}  fsdp {config.fsdp_devices}  steps {config.num_train_steps}"
    )
    print(f"  assets      {config.assets_dirs}")
    print(f"  checkpoints {config.checkpoint_dir}")

    train = _load_train_module()
    if args.dry_run:
        import openpi.training.data_loader as _data_loader

        loader = _data_loader.create_data_loader(config, num_batches=1, shuffle=False)
        observation, actions = next(iter(loader))
        print(f"  actions {actions.shape} {actions.dtype}")
        print(f"  state   {observation.state.shape}")
        print(f"  images  {[(k, v.shape) for k, v in observation.images.items()]}")
        print(f"  tokens  {observation.tokenized_prompt.shape}")
        return
    train.main(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
