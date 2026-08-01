"""Get-up task metrics: standing success rate.

Tracks the fraction of environments that have achieved standing — both
feet in contact with the ground AND pelvis above target height. This is
the primary training metric for the get-up task (analogous to
Episode_Reward/track_linear_velocity for the velocity task).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def standing_success(
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str,
) -> torch.Tensor:
    """Fraction of envs that achieved standing: both feet contact + pelvis above target height.

    Matches HumanUP's success definition: CoM height above threshold with
    both feet on the ground. A per-env binary signal (1.0 = standing,
    0.0 = not) that the metrics manager averages over the episode.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    both_feet = is_contact.all(dim=1)
    h = env.scene["robot"].data.root_link_pos_w[:, 2]
    return (both_feet & (h > target_height)).float()