"""Walka constants for mjlab entity construction.

This file intentionally defers XML existence checks until env creation time,
so task listing and package import remain usable during migration.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

WALKA_XML: Path = SRC_PATH / "assets" / "robots" / "walka" / "xmls" / "walka.xml"


def _require_xml() -> Path:
    if WALKA_XML.exists():
        return WALKA_XML
    raise FileNotFoundError(
        "Walka MJCF not found. Expected: "
        f"{WALKA_XML}. Convert your IsaacLab USD asset to MuJoCo XML and place it there."
    )


def get_spec() -> mujoco.MjSpec:
    xml_path = _require_xml()
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for act in list(spec.actuators):
        spec.delete(act)
    return spec


# Per-joint-group gains, ported 1:1 from JACKBOT_CFG's actuators dict
# (walka_lab/.../jackbot.py). A single uniform actuator across all joints
# would give the knee/ankle/arm joints the hip/waist gains (200/5), which
# is wrong by 2-40x depending on the joint — keep the groups separate.
WALKA_HIP_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_waist_joint",
        ".*_roll_hip_joint",
        ".*_pitch_hip_joint",
        ".*_yaw_hip_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=300.0,
    armature=0.02,
)

WALKA_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=100.0,
    damping=20.0,
    effort_limit=80.0,
    armature=0.02,
)

WALKA_ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*pitch_ankle_joint", ".*roll_ankle_joint"),
    stiffness=50.0,
    damping=10.0,
    effort_limit=30.0,
    armature=0.02,
)

# Split from a single "arm" group because BuiltinPositionActuatorCfg only
# takes a scalar stiffness/damping, while JACKBOT_CFG gave shoulder joints
# stiffness=50 and elbow/wrist joints stiffness=10 (damping is 10 for all
# three, so that part stays merged).
WALKA_SHOULDER_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_shoulder_joint",),
    stiffness=50.0,
    damping=10.0,
    effort_limit=100.0,
    armature=0.02,
)

WALKA_ELBOW_WRIST_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_elbow_joint", ".*_wrist_joint"),
    stiffness=10.0,
    damping=10.0,
    effort_limit=100.0,
    armature=0.02,
)

WALKA_ACTION_SCALE = 0.25

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.832),
    joint_pos={
        ".*pitch_ankle_joint": -0.305,
        ".*pitch_elbow_joint": -1.047,
        ".*pitch_knee_joint": 0.785,
        ".*pitch_hip_joint": -0.209,
        ".*pitch_shoulder_joint": 0.262,
        ".*_waist_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*collision.*",),
    contype=1,
    conaffinity=1,
    condim=3,
)

WALKA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WALKA_HIP_ACTUATOR,
        WALKA_KNEE_ACTUATOR,
        WALKA_ANKLE_ACTUATOR,
        WALKA_SHOULDER_ACTUATOR,
        WALKA_ELBOW_WRIST_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_walka_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=WALKA_ARTICULATION,
    )
