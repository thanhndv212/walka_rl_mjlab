"""Get-up task rewards: multiplicative task progress, stable-base style
rewards, stand-on-feet, and conditional pose.

Ported from the composite reward design of HumanUP (RSS 2025) and HoST
(RSS 2025 Best Systems Paper Finalist), adapted for mjlab's manager-based
env architecture. v1.2 replaces the additive height+upright+body_up stack
with HoST's actual mechanism (verified against their public code,
InternRobotics/HoST): a single *multiplicative* height x orientation task
reward (``task_progress``), so a policy can't max out orientation while
ignoring height (or vice versa) the way independent additive terms allow --
this is what their code comments cite as "follow 'Learning to Get Up'", and
it's a structurally stronger anti-exploit than height-gating alone (v1.1,
now removed). See docs/get_up_task.md for the full research trail.

Key design decisions:
- task_progress: tolerance(height) * tolerance(orientation) — HoST's core
  mechanism, both must be earned together, continuously (not just above a
  gate)
- shank_vertical / feet_level: HoST's style_shank_orientation /
  style_ground_parallel — reward a stable, feet-planted crouch as the
  intermediate pose to rise from, gated to the rising phase
- hand_supported_rise / foot_advance (v1.3): reward a human-like
  hands-then-lunge get-up strategy observed in v1.2 rollouts (converging
  to face-down, arms/legs spread) -- push the torso up with both hands
  planted, then step one foot forward to pivot upright. See
  docs/get_up_task.md.
- hands_off_ground (v1.5): v1.4 (trained to 10k iterations) achieved 100%
  genuine recovery (scripts/eval_fallen_recovery.py) but empirically kept
  one hand on the ground indefinitely once standing, bent forward at the
  waist -- nothing in the v1.4 stack ever penalized that. foot_advance is
  now also capped at success_height (it had no upper height bound before),
  and this new term directly penalizes hand contact once genuinely
  standing, rather than just removing the incentive and hoping.
- stand_on_feet: binary success signal (both feet contact + height)
- stand_still_pose: conditional penalty, zeroed during rising phase
  (the HumanUP/HoST insight: style penalties must be zeroed during
  get-up or they conflict with the task reward)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _tolerance(
    x: torch.Tensor, lower: float, margin: float, value_at_margin: float = 0.1
) -> torch.Tensor:
    """1.0 for x >= lower, smooth Gaussian falloff below it over `margin`.

    Ported from dm_control's ``rewards.tolerance`` with bounds=(lower, inf)
    -- the exact shape HoST's own orientation/head_height rewards use
    (InternRobotics/HoST, legged_gym/legged_gym/envs/g1/g1_utils.py).
    """
    scale = math.sqrt(-2 * math.log(value_at_margin))
    d = (lower - x).clamp(min=0.0) / margin
    return torch.where(x >= lower, torch.ones_like(x), torch.exp(-0.5 * (d * scale) ** 2))


def task_progress(
    env: ManagerBasedRlEnv,
    height_target: float,
    height_margin: float,
    orientation_threshold: float,
    orientation_margin: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Multiplicative height x orientation progress -- HoST's core anti-
    exploit mechanism. Replaces the v1.1 additive base_height_exp +
    upright_gated + body_up_exp stack: those three were independent,
    additive terms, so a policy could max out orientation (e.g. hold the
    torso vertical from a low kneel) largely independent of height, and the
    v1.1 empirical burst test showed exactly that (Episode_Reward/upright
    near its ceiling while genuine fallen-recovery stayed at 0%, see
    docs/get_up_task.md). Multiplying the two components forces both to be
    earned together continuously -- not just above a single gate threshold --
    since either component near zero collapses the whole product.

    height_target/orientation_threshold are the point past which each
    component saturates to 1.0; margin controls the falloff width below
    that point. Per HoST's "extend to new robots" tips: height_target ~=
    75% of standing height, orientation_threshold ~= 0.99 (tight -- this
    reward wants a properly vertical torso, not just "roughly upright").
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    pg_z = asset.data.projected_gravity_b[:, 2]  # -1 == perfectly upright
    height_component = _tolerance(h, height_target, height_margin)
    orientation_component = _tolerance(-pg_z, orientation_threshold, orientation_margin)
    return height_component * orientation_component


class shank_vertical(ManagerTermBase):
    """Reward shins (knee-to-foot) oriented vertically -- HoST's
    style_shank_orientation. A crouched/kneeling base with vertical shins
    (feet planted roughly under the knees) is the stable intermediate pose
    HoST engineers a policy to pass through before standing -- one concrete
    answer to "use the knees to form a stable base first". Gated to the
    rising phase (stage_threshold) so it doesn't reward a shin angle while
    still flat on the ground; unlike v1.1's gate this doesn't need to
    prevent farming since there's no way to have "vertical shins" while
    lying down in the first place, but the gate keeps it consistent with
    the other stable-base reward below and avoids rewarding incidental
    shin angles during the fall/settle transient.

    Uses max(dim=-1) across the two legs, not mean: a human-like
    hands-then-lunge get-up (see hand_supported_rise/foot_advance below)
    puts only the *front* leg vertical while the trailing leg stays bent --
    averaging both would dilute this to below the tolerance threshold
    exactly when the front leg is doing the real work of forming the pivot.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        knee_names = cfg.params.get("knee_body_names", ("kneeL", "kneeR"))
        foot_names = cfg.params.get("foot_body_names", ("footL", "footR"))
        asset = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        knee_ids, _ = asset.find_bodies(list(knee_names), preserve_order=True)
        foot_ids, _ = asset.find_bodies(list(foot_names), preserve_order=True)
        self._knee_ids = knee_ids
        self._foot_ids = foot_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        stage_threshold: float,
        knee_body_names: tuple[str, str] = ("kneeL", "kneeR"),
        foot_body_names: tuple[str, str] = ("footL", "footR"),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del knee_body_names, foot_body_names  # consumed in __init__ only
        asset = env.scene[asset_cfg.name]
        knee_pos = asset.data.body_link_pos_w[:, self._knee_ids, :]  # (B, 2, 3)
        foot_pos = asset.data.body_link_pos_w[:, self._foot_ids, :]
        shank = knee_pos - foot_pos
        verticality = (shank[..., 2] / shank.norm(dim=-1)).max(dim=-1).values
        reward = _tolerance(verticality, lower=0.8, margin=0.1)
        h = asset.data.root_link_pos_w[:, 2]
        return reward * (h > stage_threshold).float()


class feet_level(ManagerTermBase):
    """Reward both feet at the same height -- HoST's style_ground_parallel,
    their single largest-weighted style reward (double-support stability:
    both feet flat/level forms the base to push up from, not one foot still
    tucked under). Smooth exp-decay of the height variance rather than
    HoST's binary threshold, consistent with this task's general bias
    toward dense gradients over sparse ones. Gated to the rising phase,
    same reasoning as shank_vertical above.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        foot_names = cfg.params.get("foot_body_names", ("footL", "footR"))
        asset = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        foot_ids, _ = asset.find_bodies(list(foot_names), preserve_order=True)
        self._foot_ids = foot_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        stage_threshold: float,
        var_scale: float = 50.0,
        foot_body_names: tuple[str, str] = ("footL", "footR"),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del foot_body_names  # consumed in __init__ only
        asset = env.scene[asset_cfg.name]
        foot_z = asset.data.body_link_pos_w[:, self._foot_ids, 2]  # (B, 2)
        var = foot_z.var(dim=-1)
        reward = torch.exp(-var * var_scale)
        h = asset.data.root_link_pos_w[:, 2]
        return reward * (h > stage_threshold).float()


class hand_supported_rise(ManagerTermBase):
    """Dense reward for raising the pelvis while both hands are planted on
    the ground -- the "push up from prone" phase of a human-like get-up
    sequence observed in v1.2 rollouts (converging to face-down, arms/legs
    spread): extend the arms to lift the torso before a foot steps forward
    (see foot_advance below). Mirrors height_progress's delta pattern,
    restricted to (a) both hands in contact and (b) still below
    stage_threshold, so it specifically credits this transition rather than
    generically rewarding any upward motion the way height_progress already
    does.
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
        sensor_name: str,
        stage_threshold: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        both_hands = (sensor.data.current_contact_time > 0).all(dim=1)
        h = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
        delta = torch.where(
            self._has_prev, (h - self._prev_h).clamp(min=0.0), torch.zeros_like(h)
        )
        self._prev_h = h.clone()
        self._has_prev = torch.ones_like(self._has_prev)
        return delta * both_hands.float() * (h < stage_threshold).float()


class foot_advance(ManagerTermBase):
    """Reward one foot stepping forward of the pelvis while at least one
    hand is still down -- the asymmetric lunge that bridges a
    hand-supported push-up to standing (plant hands, step one foot forward
    and under the hips, push up through that leg). "Forward" is measured in
    the pelvis's current *yaw-only* heading frame (mjlab's ``yaw_quat``),
    not a fixed world axis, since yaw is randomized at reset -- ignoring
    current roll/pitch tilt matters here because the pelvis is usually
    still tilted during this exact phase.

    Gated on hand contact AND ``h < success_height`` (v1.5 fix): v1.4's
    hand-contact-only gate had no upper height bound, so once standing at
    full height with a foot happening to satisfy the forward-offset
    condition, this term kept paying out exactly as strongly as during the
    actual rise -- an unbounded incentive to keep a hand on the ground
    indefinitely. Empirically confirmed via video (docs/get_up_task.md):
    the v1.4 policy stood on both feet but kept one arm on the ground,
    bent forward at the waist, well past genuine standing height. Capping
    this gate the same way ``feet_level``/``shank_vertical`` already are
    removes the unbounded incentive; see also ``hands_off_ground`` below
    for the direct complementary penalty.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        foot_names = cfg.params.get("foot_body_names", ("footL", "footR"))
        asset = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        foot_ids, _ = asset.find_bodies(list(foot_names), preserve_order=True)
        self._foot_ids = foot_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        hand_sensor_name: str,
        success_height: float,
        forward_target: float = 0.15,
        forward_margin: float = 0.1,
        foot_body_names: tuple[str, str] = ("footL", "footR"),
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del foot_body_names  # consumed in __init__ only
        asset = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[hand_sensor_name]
        any_hand = (sensor.data.current_contact_time > 0).any(dim=1)
        h = asset.data.root_link_pos_w[:, 2]

        heading = yaw_quat(asset.data.root_link_quat_w)  # (B, 4)
        n_feet = len(self._foot_ids)
        heading_expanded = heading.unsqueeze(1).expand(-1, n_feet, -1)
        rel = (
            asset.data.body_link_pos_w[:, self._foot_ids, :]
            - asset.data.root_link_pos_w.unsqueeze(1)
        )
        forward_offset = quat_apply_inverse(heading_expanded, rel)[..., 0]
        most_advanced = forward_offset.max(dim=-1).values
        reward = _tolerance(most_advanced, forward_target, forward_margin)
        return reward * any_hand.float() * (h < success_height).float()


def hands_off_ground(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    success_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize any hand-ground contact once above ``success_height`` --
    the direct complement of ``stand_on_feet``. v1.4's reward stack had no
    term at all that ever asked the policy to let go of the ground: nothing
    penalized a hand staying down once genuinely standing, so a "keep one
    hand down for balance" tripod strategy was free stability with no cost
    (see ``foot_advance``'s docstring for the empirical video evidence).
    This term makes that specific behavior actively costly instead of just
    removing its incentive, which is a more direct fix than hoping the
    policy stops on its own once ``foot_advance``'s gate (above) no longer
    rewards it.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    any_hand = (sensor.data.current_contact_time > 0).any(dim=1)
    h = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
    return (any_hand & (h > success_height)).float()


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
    validation roadmap" Step 1). task_progress's height component still
    saturates via a narrow tolerance shape around its target -- near-zero
    gradient anywhere far below it, same as v1.1's base_height_exp before it --
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
    which is a precursor to standing that task_progress's height component
    gives no credit for. Mirrors height_progress's previous-value pattern,
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