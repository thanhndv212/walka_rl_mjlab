"""Get-up task events: post-reset ground-clearance correction.

Not in stock mjlab. The get-up task's initial-pose distribution samples a
pelvis height *independent* of the sampled roll/pitch (see
``env_cfgs.py::walka_get_up_env_cfg``'s ``reset_base`` event). That's fine
for poses close to the "lying flat" orientations the height range was tuned
for, but a meaningful fraction of uniform roll/pitch samples land near
upright while the pelvis height is still sampled from the near-ground range
-- in that combination, the legs (still close to their default, mostly
extended, standing configuration) are driven tens of centimeters into the
ground plane. MuJoCo's contact solver then resolves that deep interpenetration
in a single explosive correction, launching the robot to >1m within a handful
of steps -- a simulator artifact, not a get-up behavior, that was confirmed
via a zero-action rollout (see debug notes) where ~10% of freshly reset envs
shot past the `too_high` termination bound within 15 steps while holding the
default pose.

``ensure_ground_clearance`` fixes this by construction rather than by
tuning the sampling ranges (which can't rule out the upright/low-z
combination without either losing "near-ground" coverage in the
non-upright case or requiring per-orientation ranges). It runs as the last
"reset" mode event, after ``reset_base``/``reset_robot_joints`` have written
the sampled pose, and shifts the root strictly upward so that no part of the
robot's collision geometry is below the ground plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ensure_ground_clearance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    clearance: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Shift the root up so no robot geom's AABB is below the ground plane.

    Must run *after* the reset events that set root pose/joint angles (event
    order follows ``cfg.events`` dict insertion order within the "reset"
    mode). Computes each geom's world-space AABB from the (already-written,
    but not yet kinematically propagated) qpos by explicitly calling
    ``env.sim.forward()`` first -- see ``ManagerBasedRlEnv.step``'s docstring
    on why derived quantities (``geom_xpos``/``geom_xmat``) lag raw qpos
    writes until forward() runs.

    Uses each geom's local-frame AABB (``mj_model.geom_aabb``: center +
    half-extent) rotated into world frame, which over-approximates the mesh's
    true lowest point -- a conservative bound, not an exact one, but cheap
    and vectorized, and erring toward *more* clearance is the safe direction
    here.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]

    # Refresh geom_xpos/geom_xmat from the qpos written by prior reset events.
    env.sim.forward()

    geom_ids = asset.indexing.geom_ids
    aabb = torch.as_tensor(
        env.sim.mj_model.geom_aabb, device=env.device, dtype=torch.float32
    )
    aabb_center = aabb[geom_ids, 0:3]  # (G, 3), local frame
    aabb_half = aabb[geom_ids, 3:6]  # (G, 3), local frame

    xpos = env.sim.data.geom_xpos[env_ids][:, geom_ids, :]  # (N, G, 3)
    xmat = env.sim.data.geom_xmat[env_ids][:, geom_ids, :, :]  # (N, G, 3, 3)
    row_z = xmat[..., 2, :]  # (N, G, 3) -- z-row of each geom's world rotation

    center_z = xpos[..., 2] + (row_z * aabb_center).sum(dim=-1)  # (N, G)
    half_extent_z = (row_z.abs() * aabb_half).sum(dim=-1)  # (N, G)
    min_z_per_geom = center_z - half_extent_z  # (N, G)
    min_z = min_z_per_geom.min(dim=1).values  # (N,)

    shift = torch.clamp(clearance - min_z, min=0.0)  # (N,) never push down
    if not torch.any(shift > 0):
        return

    cur_pos = asset.data.root_link_pos_w[env_ids].clone()
    cur_quat = asset.data.root_link_quat_w[env_ids].clone()
    cur_pos[:, 2] += shift
    asset.write_root_link_pose_to_sim(
        torch.cat([cur_pos, cur_quat], dim=-1), env_ids=env_ids
    )
