"""The `g2_arc_v1` training config, assembled outside the openpi tree.

Nothing here is pasted into `train/openpi/`. `register.py` imports this module and injects the
TrainConfig into openpi's registry at run time, so the vendored tree stays byte-identical.

Two things differ from the G2 pack's config:

  * The action labels come from the dataset's `action_chunk` column rather than from LeRobot's
    `delta_timestamps` machinery. Setting `action_sequence_keys` to the empty tuple leaves the
    delta_timestamps dict empty (see data_loader.create_torch_dataset), so LeRobot returns one row
    per frame and the pre-computed arc-length chunk is used verbatim.
  * Prompt dropout is available but OFF. It was useful when the two task strings differed by a fact
    the head camera also shows -- how many workpieces are left -- so a sample stripped of its
    instruction was still answerable. It is harmful across pick and place: both contain stretches of
    "cup holding a part, arm moving", and only the prompt says whether the next move is to lift or to
    set down. Dropping it there manufactures genuinely ambiguous samples, which is the failure this
    whole line of work is trying to avoid.
"""

from __future__ import annotations

import dataclasses
import pathlib
import random

import agibot_g2_arc_policy
import numpy as np
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.shared.normalize as _normalize  # noqa: F401  (imported for side-effect parity)
from openpi.training import config as _config
import openpi.transforms as _transforms
from typing_extensions import override

REPO_ID = "world-studio/agibot-g2-pickplace-arc-v4"
CONFIG_NAME = "g2_pickplace_v1"
ACTION_HORIZON = 50
ACTION_DIM = 24
NEUTRAL_PROMPT = "Pick the workpieces out of the material bin with the suction cups."
PROMPT_DROPOUT = 0.0


@dataclasses.dataclass(frozen=True)
class UnflattenActionChunk(_transforms.DataTransformFn):
    """Reshape the flat per-frame chunk column back to (horizon, action_dim).

    The column is stored flat because LeRobot's parquet path is only reliable for 1-D numeric
    features; the shape is restored here, before anything else looks at it.
    """

    horizon: int
    action_dim: int
    key: str = "action_chunk"

    def __call__(self, data: dict) -> dict:
        chunk = np.asarray(data[self.key], dtype=np.float32)
        return {**data, self.key: chunk.reshape(self.horizon, self.action_dim)}


@dataclasses.dataclass(frozen=True)
class PromptDropout(_transforms.DataTransformFn):
    """Replace the episode's prompt with a neutral one on a fraction of samples.

    Uses the `random` module rather than numpy: torch's DataLoader reseeds `random` per worker, so
    the draw differs across workers and across epochs. A neutral prompt, never a wrong one -- an
    incorrect instruction would teach the policy to ignore instructions.
    """

    neutral: str
    probability: float

    def __call__(self, data: dict) -> dict:
        if self.probability <= 0.0 or "prompt" not in data or random.random() >= self.probability:
            return data
        return {**data, "prompt": self.neutral}


@dataclasses.dataclass(frozen=True)
class LeRobotAgiBotG2ArcDataConfig(_config.DataConfigFactory):
    """Data config for the arc-resampled G2 dataset.

    28-dim state / 24-dim action, three cameras, absolute joint targets so no delta transform. The
    prompt comes from the dataset task string.
    """

    horizon: int = ACTION_HORIZON
    action_dim: int = ACTION_DIM
    prompt_dropout: float = PROMPT_DROPOUT

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                PromptDropout(neutral=NEUTRAL_PROMPT, probability=self.prompt_dropout),
                UnflattenActionChunk(horizon=self.horizon, action_dim=self.action_dim),
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/left_wrist": "left_wrist_image",
                        "observation/right_wrist": "right_wrist_image",
                        "observation/state": "state",
                        # The arc-length chunk replaces the per-frame action column entirely.
                        "actions": "action_chunk",
                        "action_loss_weight": "action_loss_weight",
                        # PromptFromLeRobotTask injects "prompt" upstream, and RepackTransform keeps
                        # only the keys listed here.
                        "prompt": "prompt",
                    }
                ),
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[agibot_g2_arc_policy.AgiBotG2Inputs(model_type=model_config.model_type)],
            outputs=[agibot_g2_arc_policy.AgiBotG2Outputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def build(
    root: pathlib.Path,
    *,
    exp_name: str = "run1",
    num_train_steps: int = 30_000,
    fsdp_devices: int = 2,
    save_interval: int = 10_000,
) -> _config.TrainConfig:
    """The TrainConfig. Assets and checkpoints stay under `root`, never in train/openpi/."""
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        # Recommended capacity split from the pack, section 6: SigLIP and the action expert train in
        # full, the Gemma-2B LLM is LoRA'd. The instruction set is two fixed strings, so the language
        # model needs the least of the capacity.
        paligemma_variant="gemma_2b_lora",
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=model,
        data=LeRobotAgiBotG2ArcDataConfig(
            repo_id=REPO_ID,
            base_config=_config.DataConfig(
                prompt_from_task=True,
                # Empty on purpose: the chunk is a column, not a time window.
                action_sequence_keys=(),
            ),
        ),
        weight_loader=_config.weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=ACTION_HORIZON, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        num_train_steps=num_train_steps,
        batch_size=32,
        # Shard the model across the two devices rather than replicating it, matching the G1 run
        # that trained the same recipe on 2xH100 at batch 32 and 1.0 it/s.
        fsdp_devices=fsdp_devices,
        # Each sample decodes three video streams, so this is the throughput knob. At 8 workers the
        # first run sat at 3.3 s/it with every worker pegged and the GPUs idle; the box has 96 cores.
        num_workers=24,
        # 13 GiB a piece. Saving and keeping on the same 10k period leaves exactly 10k/20k/30k
        # rather than the six the 1k/5k defaults produce.
        save_interval=save_interval,
        keep_period=save_interval,
        ema_decay=None,  # LoRA finetuning: EMA off
        assets_base_dir=str(root / "assets"),
        checkpoint_base_dir=str(root / "checkpoints"),
    )


def register(root: pathlib.Path, **kwargs) -> _config.TrainConfig:
    """Inject the config into openpi's registry so get_config() and the asset paths resolve."""
    config = build(root, **kwargs)
    _config._CONFIGS_DICT[config.name] = config  # noqa: SLF001
    if config not in _config._CONFIGS:  # noqa: SLF001
        _config._CONFIGS.append(config)  # noqa: SLF001
    return config
