"""Get-up task rewards: height, upright, stand-on-feet, and conditional pose.

Ported from the composite reward design of HumanUP (RSS 2025) and HoST
(RSS 2025 Best Systems Paper Finalist), adapted for mjlab's manager-based
env architecture. The core idea: a set of task rewards (height + upright +
stand-on-feet) provides the primary "get up" signal, while a conditional
pose penalty only activates when near standing height to avoid conflicting
with the rising motion.

Key design decisions (from research, see .slim/deepwork/get-up-task.md):
- base_height_exp: exp(-|h - h_target|² / std²) — smooth height reward
- stand_on_feet: binary success signal (both feet contact + height)
- body_up_exp: clamped projected gravity z — upright orientation
- stand_still_pose: conditional penalty, zeroed during rising phase
  (the HumanUP/HoST insight: style penalties must be zeroed during
  get-up or they conflict with the task reward)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height_exp(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exponential reward for reaching target pelvis height.

    exp(-|h - h_target|² / std²). Saturates at 1.0 when at target height,
    decays smoothly as height deviates. This is the primary task signal
    from HumanUP/HoST — the robot must raise its pelvis to standing height.
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(h - target_height) / std**2)


def stand_on_feet(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary reward: 1.0 when both feet contact ground AND pelvis above target height.

    This is the "success" signal — the robot is standing on both feet at
    sufficient height. Prevents reward hacking where the robot achieves
    height without actually standing on its feet (e.g. propping on knees
    or hands). Mirrors HumanUP's stand_on_feet term.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    both_feet = is_contact.all(dim=1)
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    tall_enough = h > target_height
    return (both_feet & tall_enough).float()


def body_up_exp(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for torso being upright. Clamped projected gravity z.

    Returns clamp(-projected_gravity_b[:, 2], 0, 1): 1.0 when perfectly
    upright (gravity points straight down in body frame, pg_z=-1), 0.0
    when sideways or upside down. Smooth, bounded, no hyperparameters.

    This is the orientation complement to base_height_exp — height alone
    doesn't distinguish kneeling from standing (the "kneeling trap"
    pitfall flagged in the research). body_up_exp ensures the torso is
    actually vertical, not just high.
    """
    asset = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b
    return torch.clamp(-pg[:, 2], min=0.0, max=1.0)


def stand_still_pose(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Conditional pose penalty: penalize joint deviation from default, scaled by proximity to standing height.

    Scale = exp(-|h - target|² / std²): ~1 when near standing (penalize
    fidgeting), ~0 when far from standing (don't penalize the get-up
    motion itself). This is the key insight from HumanUP/HoST: style
    penalties must be zeroed during the rising phase or they conflict
    with the task reward. "Directly introducing regularization terms for
    control effort and motion speed leads to failure to learn" (Learning
    to Get Up, SIGGRAPH 2022 ablation).
    """
    asset = env.scene[asset_cfg.name]
    diff = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    penalty = torch.sum(torch.square(diff), dim=1)
    h = asset.data.root_link_pos_w[:, 2]
    scale = torch.exp(-torch.square(h - target_height) / std**2)
    return penalty * scale