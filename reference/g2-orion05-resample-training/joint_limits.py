"""AgiBot G2 joint limits, keyed by the names the LeRobot packages use.

Source: GDK knowledge base `reference/Python/robot/README.md`, the "关节限位值说明"
tables under `joint_servo_control` and `move_arm_joint_servo`. Values are radians.

The mapping from GDK's `idxNN_*` names to the dataset's names is positional and
follows the order the packages declare in `meta/info.json`: five lower-body joints,
three head joints, then seven per arm.

`joint4` is asymmetric on both arms; do not assume these ranges are centred on zero.
"""

from __future__ import annotations

from typing import Final

# name -> (minimum, maximum)
JOINT_LIMITS: Final[dict[str, tuple[float, float]]] = {
    # idx01..idx05_body_joint
    "ankle_pitch": (-1.082104, 0.000174),
    "knee_pitch": (-0.000174, 2.652900),
    "hip_pitch": (-1.919862, 1.570970),
    "hip_roll": (-0.436332, 0.436332),
    "hip_yaw": (-3.045599, 3.045599),
    # idx11..idx13_head_joint
    "head_yaw": (-1.570970, 1.570970),
    "head_roll": (-0.349240, 0.349240),
    "head_pitch": (-0.534773, 0.534773),
}

_ARM: Final[tuple[tuple[float, float], ...]] = (
    (-3.071796, 3.071796),
    (-2.059505, 2.059505),
    (-3.071796, 3.071796),
    (-2.495838, 1.012308),
    (-3.071796, 3.071796),
    (-1.012308, 1.012308),
    (-1.535907, 1.535907),
)

for _index, _bounds in enumerate(_ARM, start=1):
    JOINT_LIMITS[f"left_arm_joint_{_index}"] = _bounds
    JOINT_LIMITS[f"right_arm_joint_{_index}"] = _bounds

LIMIT_RANGE: Final[dict[str, float]] = {
    name: upper - lower for name, (lower, upper) in JOINT_LIMITS.items()
}
