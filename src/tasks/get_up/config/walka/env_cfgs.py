"""Walka get-up environment configuration for mjlab.

Builds a ManagerBasedRlEnvCfg from scratch (not derived from the velocity
template) because the get-up task is fundamentally different from
locomotion: no velocity command, no terrain curriculum, no gait clock,
and the robot starts in a fallen pose rather than standing. The reward
structure follows the composite design from HumanUP (RSS 2025) and HoST
(RSS 2025 Best Systems Paper Finalist) — see docs/get_up_task.md and
.slim/deepwork/get-up-task.md for the research basis.

Key design decisions:
- Initial poses: randomized near-ground pelvis height + full roll/pitch
  range (supine/prone/side) + randomized joint angles (±0.5 rad from
  default). No pre-generated fall trajectories — single distribution.
- Rewards: base_height_exp (primary) + upright + stand_on_feet (success
  signal) + body_up_exp (orientation) + conditional stand_still_pose
  (zeroed during rising) + regularization penalties.
- Terminations: height bounds only (no bad_orientation — the robot starts
  fallen). Too low (<0.05m) = collapsed, too high (>1.2m) = exploiting.
- No curriculum for v1; standing-probability curriculum can be added
  later if single-distribution training gets stuck (research warns this
  is a risk — the "kneeling trap" local minimum).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots import WALKA_ACTION_SCALE, get_walka_robot_cfg
from src.tasks.get_up import mdp


def walka_get_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_walka_robot_cfg()},
            num_envs=1,
            extent=2.0,
        ),
        sim=SimulationCfg(
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
            ),
            contact_sensor_maxmatch=500,
            nconmax=64,
        ),
        decimation=4,
        episode_length_s=20.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
    )

    site_names = ("footL", "footR")

    # --- Sensors ---
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(footL|footR)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=False,
    )
    cfg.scene.sensors = (
        feet_ground_cfg,
        self_collision_cfg,
        body_ground_cfg,
    )

    # --- Actions ---
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=WALKA_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    # --- Commands: none (get-up has no velocity command) ---
    cfg.commands = {}

    # --- Observations ---
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "pelvis_height": ObservationTermCfg(func=mdp.pelvis_height),
        "feet_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "body_contact": ObservationTermCfg(
            func=mdp.body_contact,
            params={"sensor_name": "body_ground_contact"},
        ),
    }

    critic_terms = {
        **actor_terms,
        # Critic sees the true (unbiased) joint positions as privileged info.
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "feet_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # --- Events ---
    # Initial fallen pose: pelvis near ground (default z=0.832, offset
    # -0.75 to -0.55 → pelvis at ~0.08-0.28m), full roll/pitch range for
    # supine/prone/side, randomized yaw for heading variety, randomized
    # joint angles (±0.5 rad from default) for pose variety.
    cfg.events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.75, -0.55),
                    "yaw": (-3.14, 3.14),
                    "roll": (-3.14, 3.14),
                    "pitch": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.5, 0.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # Must run after reset_base/reset_robot_joints: overwrites a
        # curriculum-annealed fraction of resets to a near-standing pose so
        # the policy keeps experiencing (and values) actually reaching
        # standing height, instead of the "kneeling trap" local optimum a
        # 100%-fallen distribution converges to (see mdp/events.py).
        "reset_to_standing_curriculum": EventTermCfg(
            func=mdp.reset_to_standing_curriculum,
            mode="reset",
            params={
                "start_prob": 0.5,
                "end_prob": 0.05,
                "anneal_steps": 150_000,
                "joint_noise": 0.05,
            },
        ),
        # Must run last: the full roll/pitch range combined with a pelvis
        # height tuned for lying-flat poses lets near-upright samples spawn
        # with legs driven deep into the ground plane, which MuJoCo's
        # contact solver then resolves as an explosive launch instead of a
        # real get-up trajectory. See mdp/events.py.
        "ensure_ground_clearance": EventTermCfg(
            func=mdp.ensure_ground_clearance,
            mode="reset",
            params={"clearance": 0.02},
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    geom_names=("footL_collision", "footR_collision"),
                ),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
    }

    # --- Rewards ---
    # Task rewards (primary get-up signal):
    #   base_height_exp — reach standing pelvis height (weight 5.0)
    #   upright — keep pelvis level (weight 1.0)
    #   stand_on_feet — binary success: both feet contact + height (weight 2.5)
    #   body_up — torso upright orientation (weight 0.25)
    #   height_progress / feet_force_progress — dense HumanUP-style Δ-progress
    #   reward (docs/get_up_task.md Step 1), added to give gradient far from
    #   target_height where base_height_exp's narrow Gaussian is ~flat —
    #   the exact condition the kneeling trap exploited.
    # Conditional style (zeroed during rising, active near standing):
    #   stand_still_pose — penalize joint deviation from default (weight -0.5)
    # Regularization:
    #   dof_pos_limits, action_rate_l2, self_collisions, joint_vel, torques
    # Termination penalty:
    #   is_terminated — strong penalty for falling/terminating (weight -500)
    cfg.rewards = {
        "base_height": RewardTermCfg(
            func=mdp.base_height_exp,
            weight=5.0,
            params={"target_height": 0.832, "std": 0.1},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
            },
        ),
        "stand_on_feet": RewardTermCfg(
            func=mdp.stand_on_feet,
            weight=2.5,
            params={"sensor_name": "feet_ground_contact", "target_height": 0.7},
        ),
        "body_up": RewardTermCfg(
            func=mdp.body_up_exp,
            weight=0.25,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
        ),
        "height_progress": RewardTermCfg(func=mdp.height_progress, weight=2.0),
        "feet_force_progress": RewardTermCfg(
            func=mdp.feet_force_progress,
            weight=1.0,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
        "joint_vel_penalty": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.0001,
        ),
        "torques_penalty": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-0.000001,
        ),
        "stand_still_pose": RewardTermCfg(
            func=mdp.stand_still_pose,
            weight=-0.5,
            params={
                "target_height": 0.7,
                "std": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "termination": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-500.0,
        ),
    }

    # --- Terminations ---
    # No bad_orientation (robot starts fallen — orientation is bad by design).
    # Height bounds: too low = collapsed, too high = exploiting/jumping.
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "too_low": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.05},
        ),
        "too_high": TerminationTermCfg(
            func=mdp.root_height_above_maximum,
            params={"maximum_height": 1.2},
        ),
    }

    # --- Curriculum: none for v1 ---
    cfg.curriculum = {}

    # --- Metrics ---
    cfg.metrics = {
        "standing_success": MetricsTermCfg(
            func=mdp.standing_success,
            params={"target_height": 0.7, "sensor_name": "feet_ground_contact"},
        ),
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg