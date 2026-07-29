"""Walka velocity environment configuration for mjlab."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from src.assets.robots import WALKA_ACTION_SCALE, get_walka_robot_cfg
from src.tasks.velocity import mdp


def walka_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 64

    cfg.scene.entities = {"robot": get_walka_robot_cfg()}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    site_names = ("footL", "footR")

    # Wire foot height scan to per-foot sites (base template leaves this
    # unset — "Set per-robot: frame and pattern" — so foot_height_scan has
    # no frames until this runs; foot_clearance/foot_swing_height rewards
    # and the "foot_height" critic observation all read from this sensor).
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

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
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = WALKA_ACTION_SCALE

    cfg.viewer.body_name = "pelvis"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.9
    twist_cmd.ranges.lin_vel_x = (-0.2, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    # foot_friction/base_com events are no-ops ("Set per-robot" in the base
    # template) until the real geom/body names are known — now that
    # walka.xml exists, geoms are named "<body>_collision" (see
    # build_mjcf.py) and pelvis is the floating-base body.
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "footL_collision",
        "footR_collision",
    )

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

    # "pose" (variable_posture) requires non-empty per-joint stds — the base
    # template leaves them as {} placeholders, which crashes at the first
    # reward computation (0-length std tensor vs. 26 joints). Values below
    # follow g1's tuning rationale (tight ankle-roll/waist for balance, loose
    # knee/wrist for freedom of motion) adapted to walka's own joint-name
    # convention (axis_segment_joint, e.g. "pitch_hip_joint"); these are a
    # reasonable starting point, not numbers validated on walka specifically.
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*pitch_hip_joint": 0.3,
        r".*roll_hip_joint": 0.15,
        r".*yaw_hip_joint": 0.15,
        r".*pitch_knee_joint": 0.35,
        r".*pitch_ankle_joint": 0.25,
        r".*roll_ankle_joint": 0.1,
        r"pitch_waist_joint": 0.1,
        r"yaw_waist_joint": 0.2,
        r".*pitch_shoulder_joint": 0.15,
        r".*roll_shoulder_joint": 0.15,
        r".*yaw_shoulder_joint": 0.1,
        r".*pitch_elbow_joint": 0.15,
        r".*yaw_wrist_joint": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*pitch_hip_joint": 0.5,
        r".*roll_hip_joint": 0.2,
        r".*yaw_hip_joint": 0.2,
        r".*pitch_knee_joint": 0.6,
        r".*pitch_ankle_joint": 0.35,
        r".*roll_ankle_joint": 0.15,
        r"pitch_waist_joint": 0.2,
        r"yaw_waist_joint": 0.3,
        r".*pitch_shoulder_joint": 0.5,
        r".*roll_shoulder_joint": 0.2,
        r".*yaw_shoulder_joint": 0.15,
        r".*pitch_elbow_joint": 0.35,
        r".*yaw_wrist_joint": 0.3,
    }
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Gait-clock phase observation + matching reward terms, ported from
    # unitree_rl_mjlab (src/tasks/velocity/mdp): stock mjlab's template has
    # no gait-phase concept at all, which the README's "Known gaps" section
    # flagged relative to walka_lab's bespoke IsaacLab gait reward system.
    # feet_gait is a lighter stand-in (fixed sin-clock schedule of which
    # foot should be in stance vs. swing, rewarded against measured contact)
    # rather than a port of walka_lab's full clock/coordination/symmetry
    # reward set. offset=[0.0, 0.5] assumes feet_ground_contact's primary
    # match order is (footL, footR) per its pattern above, i.e. the two feet
    # alternate stance/swing half a cycle apart.
    gait_period = 0.6
    for group in ("actor", "critic"):
        cfg.observations[group].terms["phase"] = ObservationTermCfg(
            func=mdp.phase,
            params={"period": gait_period, "command_name": "twist"},
        )
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": gait_period,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if (
            cfg.scene.terrain is not None
            and cfg.scene.terrain.terrain_generator is not None
        ):
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def walka_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = walka_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
        twist_cmd.ranges.ang_vel_z = (-0.1, 0.1)

    return cfg
