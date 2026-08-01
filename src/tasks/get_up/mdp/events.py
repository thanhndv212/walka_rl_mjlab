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
from mjlab.utils.lab_api.math import sample_uniform

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


def reset_to_standing_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    start_prob: float = 0.5,
    end_prob: float = 0.05,
    anneal_steps: int = 150_000,
    joint_noise: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Overwrite a curriculum-annealed fraction of resets to a near-standing pose.

    A first training run with 100% fallen-pose resets converged in ~600/15001
    iterations to a "kneeling trap": the ``upright`` reward (weight 1.0)
    saturated near its max while ``base_height``/``stand_on_feet`` (weights
    5.0/2.5) stayed near zero for the remaining 10000+ iterations -- the
    policy found it could hold the torso vertical without ever raising the
    pelvis, since with every episode starting fallen it never had to
    experience (and thus never learned the value of) actually reaching
    standing height. This is the exact "kneeling trap" flagged in
    docs/get_up_task.md's Known Risks, and per the original design notes
    (.slim/deepwork/get-up-task.md, "Reconciled Design Decisions" #4), v1 was
    *meant* to include a standing-probability curriculum -- it was dropped
    during implementation (``cfg.curriculum = {}``, no mix).

    Runs after ``reset_base``/``reset_robot_joints`` (which sample the fallen
    pose unconditionally) and overwrites a Bernoulli-sampled subset of
    ``env_ids`` to the default standing pose plus small joint noise, keeping
    a steady stream of high-reward "actually standing" experience in the
    replay so the policy's value function doesn't lose track of the real
    objective. ``start_prob``/``end_prob``/``anneal_steps`` anneal the mix
    (measured via ``env.common_step_counter``, incremented once per
    ``env.step()``) from mostly-standing early -- so the policy first learns
    what success looks like and how to hold it -- down to mostly-fallen, so
    the bulk of training still targets recovery from a fall. ``end_prob`` is
    kept above zero for the rest of training rather than annealed to 0, so
    the "how to stand still" signal never fully disappears.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]

    progress = min(1.0, env.common_step_counter / anneal_steps)
    prob = start_prob + (end_prob - start_prob) * progress

    is_standing = torch.rand(len(env_ids), device=env.device) < prob
    standing_ids = env_ids[is_standing]
    if len(standing_ids) == 0:
        return

    default_root_state = asset.data.default_root_state[standing_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[standing_ids]
    asset.write_root_state_to_sim(default_root_state, env_ids=standing_ids)

    default_joint_pos = asset.data.default_joint_pos[standing_ids].clone()
    noise = sample_uniform(
        -joint_noise, joint_noise, default_joint_pos.shape, env.device
    )
    joint_pos = default_joint_pos + noise
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=standing_ids)
