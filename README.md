# Walka RL Mjlab

MuJoCo/mjlab counterpart of the IsaacLab-based Walka project in this repository.

This package mirrors the `unitree_rl_mjlab` structure on purpose so conversion from
IsaacLab workflows stays smooth:

- `scripts/train.py TASK_ID ...`
- `scripts/play.py TASK_ID ...`
- `scripts/list_envs.py`
- task registration through `src/tasks/**/config/*/__init__.py`

## Current status

- Velocity tasks are scaffolded and registered, and the config now builds and
  imports cleanly (`uv run python scripts/list_envs.py` lists `Walka-Rough` /
  `Walka-Flat`):
  - Per-joint-group actuator gains (hip/waist, knee, ankle, shoulder,
    elbow+wrist) ported 1:1 from `JACKBOT_CFG` in walka_lab.
  - `foot_height_scan` sensor and `base_com` event wired per-robot; the
    `upright` reward key fixed (was referencing a nonexistent
    `body_orientation_l2` key and crashing at config-build time).
- Robot config is wired through `src/assets/robots/walka/walka_constants.py`.
- **MJCF asset is generated**, from a URDF + OBJ mesh export (not from the
  USD directly — see below): `src/assets/robots/walka/xmls/walka.xml` +
  `xmls/assets/*.obj`, produced by `tools/convert_urdf_to_mjcf.py`. Both
  `Walka-Rough` and `Walka-Flat` build, reset, and step cleanly (verified
  with 100 steps of random actions, no NaNs, stable standing height).
  - `foot_friction`/`base_com` DR events wired to the real geom/body names
    now that they exist; `pose` reward per-joint stds filled in (walka's own
    joint-name convention, tuning rationale ported from the g1 example —
    a starting point, not numbers validated on walka specifically).

### Known gaps

- **MJCF fidelity is best-effort, not CAD-verified**: the source URDF has no
  `<inertial>` tags, so body masses/inertias (~28 kg total) come from
  MuJoCo's default-density auto-computation from mesh volume, not real
  material properties. Collision geometry is the visual mesh itself (the
  URDF's `<collision>` and `<visual>` tags reference the same OBJ per link) —
  no simplified collision hulls exist yet, which is fine for getting an env
  running but expensive/less robust for real training. The IMU site has no
  known real mounting offset (placed at the pelvis origin).
- **3 body pairs need `<contact><exclude>`**: `abdomen`/`pelvis`,
  `pelvis`/`pelvisL`, `pelvis`/`pelvisR` structurally overlap at every joint
  angle (found by sweeping all 26 joints through their full range — see
  `STRUCTURAL_OVERLAP_PAIRS` in `tools/convert_urdf_to_mjcf.py`), same as
  `g1.xml`'s own short exclude list. Already excluded in the generated
  `walka.xml`; re-sweep if the mesh export changes. Other body pairs only
  touch near a joint's extreme (e.g. a leg crossing into the other leg) —
  that's genuine self-collision, left alone on purpose.
- **Gait-phase reward system**: the actual tuned jackbot task
  (`JackbotRoughEnvCfg`/`JackbotFlatEnvCfg` in walka_lab, registered as
  `Isaac-Velocity-Rough/Flat-Jackbot-v0`) uses a bespoke clock-based biped
  gait reward (`feet_clock_frc`, `feet_clock_vel`, `leg_coordination_reward`,
  `gait_symmetry_reward`, `step_length_reward`, etc., all driven by a custom
  `BipedalManagerBasedRLEnv.gait_phase` sine-wave stance/swing property).
  None of this exists here yet — `env_cfgs.py` uses mjlab's generic built-in
  velocity template (`variable_posture`, `upright`, `feet_clearance`, ...)
  instead, which is a materially different reward design, not just a
  physics-backend retune.

## Install

```bash
cd walka_rl_mjlab
make sync-cpu   # or: uv sync --extra cpu --group dev
make sync       # GPU box, CUDA 12.8: uv sync --extra cu128 --group dev
```

## Usage

```bash
uv run python scripts/list_envs.py
uv run python scripts/train.py Walka-Flat --env.scene.num-envs=4096
uv run python scripts/play.py Walka-Flat --checkpoint-file logs/rsl_rl/walka_velocity/DATE/model_1000.pt
```

## Regenerating the MJCF

`tools/convert_urdf_to_mjcf.py` takes a URDF + OBJ mesh export directory
(the geometric counterpart of the USD used in walka_lab) and writes
`src/assets/robots/walka/xmls/walka.xml` + meshes. Re-run it if a newer
mesh/URDF export shows up:

```bash
uv run python tools/convert_urdf_to_mjcf.py /path/to/v1_newjointlim
```

See the module docstring for exactly what it adds beyond a literal URDF
import (freejoint, collision-geom naming, foot sites, IMU sensors).

## Migration notes

1. Keep existing IsaacLab tasks untouched during early migration.
2. Port one environment config at a time into `src/tasks/velocity/config/walka/env_cfgs.py`.
3. Once MJCF parity is validated, retune only reward/action scales that are physics-backend sensitive.
