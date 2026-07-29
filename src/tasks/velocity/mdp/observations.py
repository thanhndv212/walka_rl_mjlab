"""Gait-clock phase observation, ported from unitree_rl_mjlab's local
src/tasks/velocity/mdp/observations.py::phase.

Not present in stock mjlab.tasks.velocity.mdp. Gives the policy a
sin/cos clock signal synced to the target gait period so foot_gait (see
rewards.py) has something to phase-lock against, and standing envs get
a flat zero signal instead of a moving clock they have no reason to
track.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = (
        torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    )
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
