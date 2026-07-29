"""Weld Walka's pelvis to the world and open an interactive viewer with one
position-actuator slider per joint, to inspect joint ranges/motion without
the robot needing to balance.

    uv run python tools/view_fixed_base.py

Reuses walka.xml as committed — including the <contact><exclude> entries for
the body pairs whose meshes structurally overlap at every joint angle (see
tools/convert_urdf_to_mjcf.py) — so collision stays on for everything else;
a leg swinging into the other leg at an extreme joint angle is genuine
self-collision, not a mesh artifact, and worth seeing while testing.

Actuator gains are imported directly from walka_constants.py (not
duplicated here) so this can't drift out of sync with the real per-joint-
group values used for training.

Sets integrator="implicitfast" (mjlab.sim.MujocoCfg's own default) rather
than leaving MuJoCo's plain default ("Euler") in place. Skipping this the
first time round made the kp=200 hip/waist position actuators numerically
explode (qvel -> hundreds of rad/s) even with collision fully disabled —
a test-harness bug, not a real instability in the asset or the actual
mjlab training pipeline, which always sets this explicitly.
"""

import re

import mjlab.actuator  # noqa: F401  (primes import order, avoids circular import)
import mujoco
from mjlab.utils.spec import create_position_actuator

from src.assets.robots.walka.walka_constants import (
    WALKA_ANKLE_ACTUATOR,
    WALKA_ELBOW_WRIST_ACTUATOR,
    WALKA_HIP_ACTUATOR,
    WALKA_KNEE_ACTUATOR,
    WALKA_SHOULDER_ACTUATOR,
    WALKA_XML,
)

ACTUATOR_GROUPS = (
    WALKA_HIP_ACTUATOR,
    WALKA_KNEE_ACTUATOR,
    WALKA_ANKLE_ACTUATOR,
    WALKA_SHOULDER_ACTUATOR,
    WALKA_ELBOW_WRIST_ACTUATOR,
)


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(WALKA_XML))

    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):
        spec.delete(j)  # weld pelvis to the world

    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.005
    spec.option.iterations = 10
    spec.option.ls_iterations = 20

    for j in spec.joints:
        for cfg in ACTUATOR_GROUPS:
            if any(re.fullmatch(p, j.name) for p in cfg.target_names_expr):
                create_position_actuator(
                    spec,
                    j.name,
                    stiffness=cfg.stiffness,
                    damping=cfg.damping,
                    effort_limit=cfg.effort_limit,
                    armature=cfg.armature,
                )
                break
        else:
            raise ValueError(f"No actuator group matched joint: {j.name}")

    model = spec.compile()
    return model, mujoco.MjData(model)


if __name__ == "__main__":
    import mujoco.viewer

    model, data = build_model()
    mujoco.viewer.launch(model, data)
