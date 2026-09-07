"""Turn the policy's joint-space error into millimetres at the suction cup, using real kinematics.

Everything before this was joint-space: "0.03 rad over 22 dims" does not say whether the cup lands on
the workpiece. Converting needs forward kinematics, which needs the link geometry -- so this reads
AgiBot's published G2 URDF and walks the chain itself. No simulator, no FK library: a serial chain is
a product of fixed transforms and one rotation per joint, and writing the 40 lines keeps the joint
mapping visible instead of buried in a solver's naming conventions.

The mapping was verified rather than assumed. The URDF's body and head joints carry axes that match
our channel names one for one -- ankle/knee/hip are pitch about Y, hip_roll is X, hip_yaw is Z, the
head is Z/X/Y -- and the counts line up exactly: 5 body + 3 head + 7 + 7 = our 22 action dims, in the
same order.

Two caveats travel with the number:

  * The published URDF is the omnipicker variant. The arm chain is identical for ours; only the tool
    transform past `arm_*_end_joint` differs, and ours is a suction cup. The gripper's tool centre is
    used as a stand-in, so displacement caused by the last two joints is approximate. Reporting the
    flange as well brackets how much that matters.
  * Our robot has its legs removed and sits on a lift, so the body chain is the intended kinematics
    rather than the physical one. The arm-only figure is therefore the one to trust; the full-chain
    figure is an upper bound that assumes the body joints move the torso exactly as the URDF says.

  cd tmp/tmp_train_G2 && ../../train/openpi/.venv/bin/python eef_error.py --urdf assets_urdf/G2_omnipicker_fixed_dual.urdf
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import progress_resample as ar  # noqa: E402
import tyro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

# our action dim -> URDF joint name. Verified by axis and by count; see the module docstring.
DIM_TO_JOINT: dict[int, str] = {
    0: "idx01_body_joint1", 1: "idx02_body_joint2", 2: "idx03_body_joint3",
    3: "idx04_body_joint4", 4: "idx05_body_joint5",
    5: "idx11_head_joint1", 6: "idx12_head_joint2", 7: "idx13_head_joint3",
    **{7 + i: f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)},
    **{14 + i: f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)},
}  # fmt: skip
TIPS = {"right": "gripper_r_center_link", "left": "gripper_l_center_link"}
FLANGES = {"right": "arm_r_end_link", "left": "arm_l_end_link"}


@dataclasses.dataclass(frozen=True)
class Args:
    urdf: pathlib.Path = pathlib.Path(__file__).resolve().parent / "assets_urdf" / "G2_omnipicker_fixed_dual.urdf"
    arrays: pathlib.Path = ROOT / "out" / "closed_loop_arrays.npz"
    out: pathlib.Path = ROOT / "out" / "eef_error.json"


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])  # fmt: skip


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues. The URDF gives every revolute joint a unit axis in its own frame."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclasses.dataclass(frozen=True)
class Link:
    """One step along a chain: a fixed transform, then a rotation about `axis` by the joint value."""

    name: str
    translation: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray | None  # None for a fixed joint


def build_chain(urdf: pathlib.Path, tip: str) -> list[Link]:
    root = ET.parse(urdf).getroot()
    joints = [j for j in root.iter("joint") if j.find("child") is not None]
    by_child = {j.find("child").get("link"): j for j in joints}

    ordered, link, seen = [], tip, set()
    while link in by_child and link not in seen:
        seen.add(link)
        ordered.append(by_child[link])
        link = by_child[link].find("parent").get("link")

    chain = []
    for joint in reversed(ordered):
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()]) if origin is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()]) if origin is not None else np.zeros(3)
        axis_el = joint.find("axis")
        axis = (
            np.array([float(v) for v in axis_el.get("xyz").split()])
            if axis_el is not None and joint.get("type") in ("revolute", "continuous")
            else None
        )
        chain.append(Link(joint.get("name"), xyz, _rpy(*rpy), axis))
    return chain


def forward(chain: list[Link], values: dict[str, float]) -> np.ndarray:
    """Position of the chain's tip, in the root frame."""
    position, rotation = np.zeros(3), np.eye(3)
    for link in chain:
        position = position + rotation @ link.translation
        rotation = rotation @ link.rotation
        if link.axis is not None:
            rotation = rotation @ _axis_rotation(link.axis, values.get(link.name, 0.0))
    return position


def main(args: Args) -> None:
    data = np.load(args.arrays)
    preds, labels, in_event = data["preds"], data["labels"], data["in_event"]
    joint_dims = list(ar.JOINT_DIMS)

    chains = {
        f"{side}_{what}": build_chain(args.urdf, name)
        for what, table in (("tip", TIPS), ("flange", FLANGES))
        for side, name in table.items()
    }
    for name, chain in chains.items():
        movable = [link.name for link in chain if link.axis is not None]
        print(f"{name:14s} {len(chain)} transforms, {len(movable)} joints: {movable[0]} .. {movable[-1]}")

    results: dict[str, dict] = {}
    for name, chain in chains.items():
        in_chain = {link.name for link in chain if link.axis is not None}
        # Only the dims that actually move this tip; the head and the far arm cannot.
        dims = [d for d in joint_dims if DIM_TO_JOINT[d] in in_chain]
        arm_dims = [d for d in dims if d >= 8]

        displacement, arm_only = [], []
        for i in range(len(preds)):
            if not in_event[i]:
                continue
            for step in range(0, 50, 4):  # every 4th step keeps the FK cost down
                q = {DIM_TO_JOINT[d]: float(labels[i][step, d]) for d in joint_dims}
                base = forward(chain, q)
                full = dict(q)
                for d in dims:
                    full[DIM_TO_JOINT[d]] = float(preds[i][step, d])
                displacement.append(float(np.linalg.norm(forward(chain, full) - base)))
                arm = dict(q)
                for d in arm_dims:
                    arm[DIM_TO_JOINT[d]] = float(preds[i][step, d])
                arm_only.append(float(np.linalg.norm(forward(chain, arm) - base)))

        results[name] = {
            "dims_in_chain": dims,
            "n": len(displacement),
            "full_chain_mm": {
                "p50": round(float(np.percentile(displacement, 50)) * 1000, 1),
                "p95": round(float(np.percentile(displacement, 95)) * 1000, 1),
                "max": round(float(np.max(displacement)) * 1000, 1),
            },
            "arm_joints_only_mm": {
                "p50": round(float(np.percentile(arm_only, 50)) * 1000, 1),
                "p95": round(float(np.percentile(arm_only, 95)) * 1000, 1),
                "max": round(float(np.max(arm_only)) * 1000, 1),
            },
        }

    print(json.dumps(results, indent=2))
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
