"""Get-up task terminations: height bounds for the get-up scenario.

Unlike the velocity task (which terminates on bad_orientation > 70°), the
get-up task starts in a fallen pose — so orientation-based termination
would kill every episode at reset. Instead, terminate only on extreme
height bounds: too low (collapsed below recoverable) or too high
(jumping/exploiting).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_above_maximum(
    env: ManagerBasedRlEnv,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the asset's root height exceeds the maximum height.

    Catches reward hacking where the robot exploits dynamics to launch
    above standing height (bouncing, jumping) to rack up height reward
    without actually standing.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > maximum_height