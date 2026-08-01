"""Get-up task observations: pelvis height and body ground contact state.

Not in stock mjlab. The get-up task needs the policy to know how high the
pelvis is (the primary progress signal) and whether the torso is touching
the ground (to distinguish "lying down" from "standing up"). These are
the minimal additions over the stock proprioceptive observation set
(base lin/ang vel, projected gravity, joint pos/vel, last action) that
the get-up literature (HumanUP, HoST, UHG) consistently includes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pelvis_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pelvis z-position in world frame. Shape (num_envs, 1).

    The primary progress signal for get-up: the policy needs to know how
    high it is to drive the base_height_exp reward's implicit gradient.
    """
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def body_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Whether the torso/pelvis is in contact with the ground. Shape (num_envs, 1).

    Distinguishes "lying on the ground" from "standing up" — critical for
    the policy to know whether it needs to push off or stabilize. Uses
    sensor.data.found (the contact-found field) rather than
    current_contact_time, which is None until the first sim step and
    breaks observation shape probing at init time. The found field for a
    single-slot sensor has shape (num_envs, num_slots) = (N, 1), so we
    reduce across slots with .any(dim=1) to get a per-env boolean.
    """
    sensor = env.scene.sensors[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    is_contact = sensor_data.found > 0
    return is_contact.any(dim=1, keepdim=True).float()