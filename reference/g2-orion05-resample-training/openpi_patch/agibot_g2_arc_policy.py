"""Transforms between the AgiBot G2 (dual-suction) schema and the model's fixed schema.

Copied from the G2 fine-tuning pack. The `action_loss_weight` branch is back: the event windows run
at recorded time, which prevents them being compressed but no longer makes them denser than their
surroundings, so sampling density alone no longer emphasises the grasp.


State (28) and actions (24) are asymmetric: the four vacuum-feedback dims (left/right ch1, ch2) are
observations only and have no action counterpart. Both vectors are zero-padded to the model's
32-dim action space by PadStatesAndActions, after normalisation.

  state   [28]  0-7 lower body + head, 8-14 left arm, 15-21 right arm,
                22 left_suction_enabled, 23-24 left vacuum, 25 right_suction_enabled,
                26-27 right vacuum
  actions [24]  0-21 next-frame absolute joint targets, 22-23 suction commands

Actions are absolute joint targets, so there is no delta transform.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 28
ACTION_DIM = 24  # the model emits 32; the trailing 8 are padding
IMAGE_HW = (480, 640)


def make_agibot_g2_example() -> dict:
    """Random input example for the AgiBot G2 policy (matches the inference key format)."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(*IMAGE_HW, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(*IMAGE_HW, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(*IMAGE_HW, 3), dtype=np.uint8),
        "prompt": "Pick the parts out of the bin one at a time, using a single suction arm for each part.",
    }


def _parse_image(image) -> np.ndarray:
    """Normalise an image to HWC uint8.

    LeRobot yields CHW float tensors during training; the robot sends HWC uint8 at inference.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgiBotG2Inputs(transforms.DataTransformFn):
    """Maps the repacked G2 sample onto the model's Observation schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            # All three cameras are present on every frame of this dataset.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # Per-chunk-step loss multiplier. Training only; at inference the key is absent, the field
        # becomes None and sample_actions never looks at it.
        if "action_loss_weight" in data:
            inputs["action_loss_weight"] = data["action_loss_weight"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AgiBotG2Outputs(transforms.DataTransformFn):
    """Drops the model's padding dims, returning the robot's 24-dim action."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
