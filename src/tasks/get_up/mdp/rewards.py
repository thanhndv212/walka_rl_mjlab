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
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

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
    stage_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for torso being upright. Clamped projected gravity z, height-gated.

    Returns clamp(-projected_gravity_b[:, 2], 0, 1) * (h > stage_threshold):
    1.0 when perfectly upright (gravity points straight down in body frame,
    pg_z=-1) AND above stage_threshold, 0.0 when sideways/upside down OR
    still on the ground. Smooth, bounded above the gate.

    This is the orientation complement to base_height_exp — height alone
    doesn't distinguish kneeling from standing (the "kneeling trap"
    pitfall flagged in the research). body_up_exp ensures the torso is
    actually vertical, not just high.

    The stage_threshold gate (docs/get_up_task.md Step 2) closes the
    "kneeling trap" exploit directly: without it, this term (and
    ``upright_gated``) pays full reward for a vertical torso regardless of
    height, which the empirical burst-test run showed a policy can farm
    indefinitely from a kneeling/low pose (``Episode_Reward/upright`` sat
    near its 0.95+ ceiling for the whole run while height rewards stayed
    far below theirs) -- Step 1's dense progress rewards alone weren't
    enough to outweigh that free, height-independent signal. Pick
    stage_threshold well below stand_on_feet's target_height (e.g.
    0.3-0.4m vs. 0.7m) so early genuine progress still gets rewarded; the
    goal is closing the free-reward exploit, not making the reward sparse.
    """
    asset = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b
    h = asset.data.root_link_pos_w[:, 2]
    gate = (h > stage_threshold).float()
    return torch.clamp(-pg[:, 2], min=0.0, max=1.0) * gate


def upright_gated(
    env: ManagerBasedRlEnv,
    std: float,
    stage_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Height-gated re-derivation of mjlab's stock ``upright`` reward.

    ``mjlab.tasks.velocity.mdp.rewards.upright`` is a class-based term
    (auto-instantiated by RewardManager as ``upright(cfg, env)``, not a
    plain function) -- it can't be wrapped by calling it like a function,
    so its core math (projected gravity's xy magnitude -> exp(-xy²/std²))
    is re-derived here rather than reused, then gated the same way as
    ``body_up_exp`` above (see that docstring for why the gate exists).
    """
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        body_quat_w = body_quat_w.squeeze(1)
    else:
        body_quat_w = asset.data.root_link_quat_w
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    upright = torch.exp(-xy_squared / std**2)
    h = asset.data.root_link_pos_w[:, 2]
    gate = (h > stage_threshold).float()
    return upright * gate


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


class height_progress(ManagerTermBase):
    """Dense reward for upward pelvis-height motion since the previous step.

    HumanUP's r_Δheight term (see docs/get_up_task.md, "Implementation and
    validation roadmap" Step 1). base_height_exp's narrow Gaussian around
    target_height gives near-zero gradient anywhere far from standing --
    exactly the condition the "kneeling trap" exploited, since a policy that
    never experiences useful gradient toward standing has no reason to try.
    This term pays for making upward progress from wherever the robot
    currently is, dense across the whole height range. clamp(min=0.0) pays
    only for rising -- a controlled descent shouldn't be penalized here (that
    is termination's job), but it shouldn't be paid for either.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        del cfg
        self._prev_h = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._has_prev[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        # Deliberately not seeded from root_link_pos_w in reset(): RewardManager
        # .reset() runs inside _reset_idx() before this env's own sim.forward()
        # call (in env.reset()/step()) refreshes kinematics from the
        # just-applied reset events. Relying on some other event (e.g.
        # ensure_ground_clearance, which happens to call forward() itself) to
        # have already refreshed it first would silently break if that event
        # were ever reordered or removed. Deferring the baseline capture to
        # the first real __call__ -- which always runs after the env's own
        # forward() -- sidesteps the ordering dependency entirely.
        h = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
        delta = torch.where(
            self._has_prev, (h - self._prev_h).clamp(min=0.0), torch.zeros_like(h)
        )
        self._prev_h = h.clone()
        self._has_prev = torch.ones_like(self._has_prev)
        return delta


class feet_force_progress(ManagerTermBase):
    """Dense reward for increasing vertical ground-reaction force at the feet.

    HumanUP's r_Δfeet_contact_forces term, the counterpart to
    ``height_progress`` -- see docs/get_up_task.md Step 1. Rewards
    transferring weight onto the feet even before pelvis height itself starts
    climbing (e.g. rolling from supine onto the feet before pushing up),
    which is a precursor to standing that base_height_exp gives no credit
    for. Mirrors height_progress's previous-value/clamp-to-positive pattern,
    tracking summed vertical foot force instead of height.

    ``feet_ground_contact``'s ``reduce="netforce"`` sensor reports the
    contact-normal force with the primary (foot) bearing weight on the
    secondary (terrain) as *negative* z -- confirmed empirically (a
    standing-curriculum reset settles to force_z of -700 to -1000 per foot,
    not positive) -- so the sign must be flipped before clamping to a
    positive "weight-bearing" magnitude. Silently getting this backwards
    zeroes the term outright (clamp(min=0.0) on the wrong sign always
    returns 0), which is exactly what a 150-iteration smoke run caught:
    ``Episode_Reward/feet_force_progress`` stayed at 0.0000 for the entire
    run before this fix.

    Raw force is in Newtons -- hundreds of N per foot even at rest, versus
    height_progress's O(0.01-0.05) meters per step, so it's normalized by
    the robot's total body weight (mass * gravity, read once at init from
    ``mj_model``) into a dimensionless "fraction of body weight" before
    diffing. The per-step delta is also clamped to 1.0 (at most one full
    body-weight-equivalent gained per step): the contact solver produces
    real multi-body-weight impulse spikes on touchdown (observed up to
    ~3.4x body weight in a single 20ms step during verification) that are
    solver noise, not signal, and would otherwise dominate every other
    reward term on exactly the steps a foot first lands.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        del cfg
        gravity_mag = abs(float(env.sim.mj_model.opt.gravity[2]))
        total_mass = float(env.sim.mj_model.body_mass.sum())
        self._body_weight = total_mass * gravity_mag
        self._prev_force = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._has_prev[env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        assert sensor.data.force is not None
        force = (-sensor.data.force[..., 2]).clamp(min=0.0).sum(dim=1)
        force_frac = force / self._body_weight
        delta = torch.where(
            self._has_prev,
            (force_frac - self._prev_force).clamp(min=0.0, max=1.0),
            torch.zeros_like(force_frac),
        )
        self._prev_force = force_frac.clone()
        self._has_prev = torch.ones_like(self._has_prev)
        return delta