"""Clock-based gait reward + stand-still penalty, ported from
unitree_rl_mjlab's local src/tasks/velocity/mdp/rewards.py.

Neither is in stock mjlab.tasks.velocity.mdp. This is the piece
docs/kinematic_structure_analysis.md and the README's "Known gaps"
section flagged as missing relative to walka_lab's bespoke IsaacLab
gait-phase reward system (feet_clock_frc/feet_clock_vel/etc. driven by
a custom BipedalManagerBasedRLEnv.gait_phase property) — feet_gait is a
much lighter stand-in: a fixed sin-clock schedule of which foot should
be in stance vs. swing (see observations.py::phase), rewarded against
actual contact state, rather than a learned/adaptive phase. Good enough
to shape an alternating gait; not a port of the full walka_lab system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(
        offset, device=env.device, dtype=global_phase.dtype
    ).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
