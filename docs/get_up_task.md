# Get-Up Task Design for Walka

What the `Walka-GetUp` task does, why each reward term is there, and how
the design follows from research on humanoid fall recovery (HumanUP,
HoST, Learning to Get Up, UHG). Most terms are custom
(`src/tasks/get_up/mdp/`); the stock mjlab terms used are noted inline.

## Overview

`Walka-GetUp` trains a policy to recover from fallen poses — supine
(lying face up), prone (face down), seated, kneeling, and side-lying —
to a stable standing pose. Unlike the velocity task
(`src/tasks/velocity/`), there is no velocity command to track: the
sole objective is to stand up. The robot starts each episode in a
randomized near-ground pose and must raise its pelvis to standing
height (~0.83m) while achieving an upright torso orientation and
planting both feet on the ground.

The task is built from scratch as a `ManagerBasedRlEnvCfg` (not derived
from `make_velocity_env_cfg()`) because the get-up scenario differs
fundamentally from locomotion: no velocity command, no terrain
curriculum, no gait clock, and the robot starts in a fallen pose rather
than standing upright.

## Reward design

What each of the 11 reward terms in `RewardManager` does, why it's
there, and how it shapes the get-up motion. The reward structure
follows the composite design from HumanUP (RSS 2025) and HoST (RSS 2025
Best Systems Paper Finalist): a set of task rewards provides the
primary "get up" signal, a conditional style penalty only activates
when near standing, and regularization penalties keep the motion
smooth and safe.

| Term | Weight | Kind | One-line purpose |
|---|---|---|---|
| `base_height` | 5.0 | bonus | Reach standing pelvis height |
| `upright` | 1.0 | bonus | Keep the pelvis level |
| `stand_on_feet` | 2.5 | bonus | Binary success: both feet contact + height |
| `body_up` | 0.25 | bonus | Torso upright orientation |
| `dof_pos_limits` | -1.0 | penalty | Stay inside soft joint limits |
| `action_rate_l2` | -0.1 | penalty | Smooth actions (discourage jitter) |
| `self_collisions` | -1.0 | penalty | Avoid self-contact above 10N |
| `joint_vel_penalty` | -0.0001 | penalty | Energy efficiency (low joint velocities) |
| `torques_penalty` | -1e-6 | penalty | Energy (low torques) |
| `stand_still_pose` | -0.5 | conditional | Hold default pose when near standing |
| `termination` | -500.0 | penalty | Strong penalty for terminating/falling |

### Task rewards — the actual objective

**`base_height`** (`src/tasks/get_up/mdp/rewards.py::base_height_exp`,
weight 5.0, `target_height=0.832`, `std=0.1`) is the primary task
signal: `exp(-|h - h_target|² / std²)`, where `h` is the pelvis
z-position and `h_target=0.832` is the standing height from
`walka_constants.py::INIT_STATE`. Saturates at 1.0 when the pelvis is at
standing height and decays smoothly as height deviates. This is the
single biggest reward weight — the policy's primary drive is "raise
the pelvis." The `std=0.1` makes the reward sensitive within ~10cm of
the target, so the policy gets meaningful gradient even when close.

**`upright`** (stock mjlab `upright` class, weight 1.0,
`std=√0.2`, `asset_cfg.body_names=("pelvis",)`) penalizes the pelvis's
tilt relative to world-up (projected gravity's xy component in the
pelvis frame). Same term as the velocity task — keeps the pelvis
level. This is the complement to `base_height`: height alone doesn't
distinguish kneeling from standing (the "kneeling trap" pitfall
flagged in the research), so `upright` ensures the pelvis is actually
vertical, not just high.

**`stand_on_feet`** (`src/tasks/get_up/mdp/rewards.py::stand_on_feet`,
weight 2.5, `target_height=0.7`) is the binary success signal: returns
1.0 when both feet are in contact with the ground AND the pelvis is
above 0.7m, else 0.0. This prevents reward hacking where the robot
achieves height without actually standing on its feet (e.g. propping on
knees or hands). Mirrors HumanUP's `stand_on_feet` term — the robot
must be standing on both feet at sufficient height to earn this
reward.

**`body_up`** (`src/tasks/get_up/mdp/rewards.py::body_up_exp`, weight
0.25) returns `clamp(-projected_gravity_b[:, 2], 0, 1)`: 1.0 when
perfectly upright (gravity points straight down in body frame,
`pg_z=-1`), 0.0 when sideways or upside down. This is the orientation
complement to `base_height` — a smooth, bounded, hyperparameter-free
upright signal. The small weight (0.25) makes it a shaping term rather
than a dominant signal; `upright` (weight 1.0) handles the primary
orientation penalty.

### Conditional style — the key insight from HumanUP/HoST

**`stand_still_pose`**
(`src/tasks/get_up/mdp/rewards.py::stand_still_pose`, weight -0.5,
`target_height=0.7`, `std=0.1`) penalizes joint deviation from the
default standing pose, but scaled by proximity to standing height:
`scale = exp(-|h - target|² / std²)`. The scale is ~1 when near
standing (penalize fidgeting) and ~0 when far from standing (don't
penalize the get-up motion itself).

This is the critical design insight from HumanUP and HoST: style
penalties must be zeroed during the rising phase or they conflict with
the task reward. "Directly introducing regularization terms for control
effort and motion speed leads to failure to learn" (Learning to Get Up,
SIGGRAPH 2022 ablation). Without this conditional gating, the policy
faces a contradiction: the task reward says "raise the pelvis" (which
requires large joint deviations from the default standing pose), while
the style penalty says "stay near the default pose." The conditional
scale resolves this: the style penalty only activates once the robot is
near standing, at which point holding the default pose is the right
thing to do.

### Regularization — smoothness and safety

**`dof_pos_limits`** (stock `mjlab.envs.mdp.joint_pos_limits`, weight
-1.0) penalizes crossing the soft joint limits
(`soft_joint_pos_limit_factor=0.9` in `walka_constants.py`).

**`action_rate_l2`** (stock `mjlab.envs.mdp.action_rate_l2`, weight
-0.1) penalizes step-to-step change in the raw policy output — the
standard anti-jitter term.

**`self_collisions`** (`mdp.self_collision_cost` from velocity mdp,
weight -1.0, `force_threshold=10N`) reads the `self_collision`
`ContactSensor` and counts substeps where self-contact force exceeds
10N. Same as the velocity task.

**`joint_vel_penalty`** (stock `mjlab.envs.mdp.joint_vel_l2`, weight
-0.0001) penalizes high joint velocities — energy efficiency. The tiny
weight keeps it a shaping term rather than a dominant penalty.

**`torques_penalty`** (stock `mjlab.envs.mdp.joint_torques_l2`, weight
-1e-6) penalizes high torques — energy. Even smaller weight, same
rationale.

### Termination penalty

**`termination`** (stock `mjlab.envs.mdp.is_terminated`, weight -500.0)
is a strong penalty for any termination event (too_low, too_high,
time_out). This discourages the policy from finding "solutions" that
involve terminating the episode early (e.g. collapsing to trigger
reset). The large magnitude (-500) makes it the dominant signal if the
robot is about to terminate — the policy should avoid termination at
all costs.

## Observation space

What the policy sees and why. The get-up task needs more state
information than locomotion because the robot starts in an arbitrary
fallen pose and must infer its configuration to plan a recovery.

### Actor observations (85 dims)

| Term | Dims | Source | Purpose |
|---|---|---|---|
| `base_lin_vel` | 3 | `robot/imu_lin_vel` sensor | Base linear velocity |
| `base_ang_vel` | 3 | `robot/imu_ang_vel` sensor | Base angular velocity |
| `projected_gravity` | 3 | `mdp.projected_gravity` | Gravity in body frame (orientation) |
| `joint_pos` | 24 | `mdp.joint_pos_rel` (biased) | Joint positions relative to default |
| `joint_vel` | 24 | `mdp.joint_vel_rel` | Joint velocities |
| `actions` | 24 | `mdp.last_action` | Previous actions |
| `pelvis_height` | 1 | `mdp.pelvis_height` (custom) | Pelvis z-position — primary progress signal |
| `feet_contact` | 2 | `mdp.foot_contact` | Per-foot ground contact state |
| `body_contact` | 1 | `mdp.body_contact` (custom) | Torso ground contact state |

### Critic observations (91 dims)

Same as actor, plus:
- `joint_pos` (unbiased, privileged) — 24 dims
- `feet_contact_forces` — 6 dims (per-foot force, log-scaled)

### Why these observations

**`pelvis_height`** is the primary progress signal — the policy needs
to know how high it is to drive the `base_height` reward's implicit
gradient. Without it, the policy has no direct height input (only
inferred from `projected_gravity` and `base_lin_vel`).

**`feet_contact`** and **`body_contact`** distinguish "lying on the
ground" from "standing up" — critical for the policy to know whether
it needs to push off or stabilize. The get-up literature (HumanUP,
HoST, UHG) consistently includes full-body contact state.

**No action history** is included in v1 — RSL-RL's PPO implementation
handles history via the actor's recurrent processing. If training
struggles with partial observability, explicit action history can be
added (HoST uses 6-step history, HumanUP uses 10-step).

## Initial pose distribution

How fallen poses are randomized at episode reset.

### Reset event configuration

The `reset_base` event (`mdp.reset_root_state_uniform`) randomizes the
root state:

| Parameter | Range | Effect |
|---|---|---|
| x | (-0.5, 0.5) | Random spawn position |
| y | (-0.5, 0.5) | Random spawn position |
| z | (-0.75, -0.55) | Pelvis near ground (~0.08-0.28m) |
| yaw | (-3.14, 3.14) | Random heading |
| roll | (-3.14, 3.14) | Full roll range (supine/prone/side) |
| pitch | (-3.14, 3.14) | Full pitch range (face up/down) |

The z offset is **negative** because `reset_root_state_uniform` adds
to the default root state (z=0.832). An offset of -0.75 to -0.55 places
the pelvis at ~0.08-0.28m — near the ground, lying down. The full
roll/pitch range (-π to π) covers all fallen orientations: supine (face
up), prone (face down), and side-lying.

The `reset_robot_joints` event (`mdp.reset_joints_by_offset`)
randomizes joint positions by ±0.5 rad from the default standing pose,
creating varied fallen configurations (bent knees, extended arms,
twisted torso).

### What poses this covers

- **Supine** (face up): roll ≈ 0, pitch ≈ π → pelvis up, face up
- **Prone** (face down): roll ≈ 0, pitch ≈ 0 → pelvis up, face down
- **Side-lying**: roll ≈ ±π/2 → pelvis sideways
- **Seated** (approximate): high z offset + bent-knee joint randomization
- **Kneeling** (approximate): low z offset + bent-knee joint randomization

The distribution is a single uniform sample — no pre-generated fall
trajectories, no curriculum over difficulty. This is a v1 design; the
research warns that single-distribution training risks the "kneeling
trap" local minimum (see Known Risks).

## Termination conditions

Why height bounds only, and what each catches.

| Term | Condition | Purpose |
|---|---|---|
| `time_out` | Episode length > 20s | Prevent infinite episodes |
| `too_low` | Pelvis height < 0.05m | Collapsed below recoverable |
| `too_high` | Pelvis height > 1.2m | Jumping/exploiting |

**No `bad_orientation` termination** — the velocity task terminates
when the torso tilts > 70° from vertical, but the get-up task starts
in a fallen pose where the torso is tilted by definition. An
orientation-based termination would kill every episode at reset.

The `too_low` threshold (0.05m) is very permissive — the robot can
sink low during recovery without terminating. Only a complete collapse
(pelvis on the ground) triggers it. The `too_high` threshold (1.2m)
catches reward hacking where the robot exploits dynamics to launch
above standing height (bouncing, jumping) to rack up the `base_height`
reward without actually standing.

## Training config

PPO hyperparameters via RSL-RL.

| Parameter | Value | Notes |
|---|---|---|
| Actor hidden dims | (512, 256, 128) | Same as velocity task |
| Critic hidden dims | (512, 256, 128) | Same as velocity task |
| Activation | ELU | Same as velocity task |
| Learning rate | 1e-3 | Adaptive schedule |
| Clip param | 0.2 | Standard PPO |
| Entropy coef | 0.01 | Standard exploration |
| Gamma | 0.99 | Standard discount |
| Lambda (GAE) | 0.95 | Standard GAE |
| Max iterations | 15001 | More than velocity (10001) — get-up is harder |
| Num steps per env | 24 | Same as velocity task |
| Save interval | 100 | Checkpoint every 100 iterations |
| Experiment name | `walka_get_up` | Logs to `logs/rsl_rl/walka_get_up/` |

The network architecture and most hyperparameters are identical to the
velocity task's successful config. The only change is
`max_iterations=15001` (vs 10001) — get-up is a harder exploration
problem than locomotion and may need more iterations to converge.

## Research basis

The design follows from research on RL methods for humanoid get-up and
fall recovery:

### Key papers

- **HumanUP** (RSS 2025) — Two-stage RL (discovery → refinement) for
  Unitree G1 get-up. Key insight: multiplicative task reward +
  conditional style penalties zeroed during rising. Real-world
  deployment.
- **HoST** (RSS 2025 Best Systems Paper Finalist) — Multi-critic reward
  groups, standing-probability curriculum, action rescaling for
  sim-to-real. Deployed on Unitree G1.
- **Learning to Get Up** (SIGGRAPH 2022) — Three-stage curriculum
  (discover → weaker → slower). Key finding: direct regularization
  without curriculum → FAILS. Strong-to-weak torque curriculum
  essential.
- **Unified Humanoid Get-Up** (UHG) — Pure MuJoCo + CrossQ, 5-DoF
  sagittal control. Zero-shot across 7 morphologies.
- **PHC** (ICCV 2023) — Fail-state recovery, progressive primitives.

### Key insights applied

1. **Conditional style penalties** — `stand_still_pose` is zeroed during
   the rising phase (scaled by proximity to standing height). From
   HumanUP/HoST.
2. **Height + orientation** — `base_height` alone risks the kneeling
   trap; `upright` + `body_up` ensure the torso is actually vertical.
   From Learning to Get Up ablations.
3. **Binary success signal** — `stand_on_feet` prevents reward hacking
   where height is achieved without standing on feet. From HumanUP.
4. **Strong termination penalty** — -500 weight discourages early
   termination. From HumanUP/HoST.
5. **No curriculum (v1)** — Single-distribution training is a known
   risk but simpler. Curriculum can be added if training fails.

## Known risks and future work

### Known risks

**Kneeling trap** — The research warns that single-distribution
training (no curriculum) risks the policy getting stuck in a kneeling
pose: height reward is partially satisfied, but the robot can't
progress to full standing. Mitigations if this occurs:
- Add `body_up` weight increase (currently 0.25)
- Add standing-probability curriculum (start with some envs standing)
- Add strong-to-weak torque curriculum

**No curriculum** — v1 uses a single fallen-pose distribution. The
research consistently shows that curriculum (standing-probability,
torque strength, or staged training) improves success rate. This is
the most likely v2 improvement.

**No action history** — v1 relies on RSL-RL's implicit history
handling. If partial observability hurts, explicit action history
(6-10 steps, as in HoST/HumanUP) can be added.

**Sim-to-real gap** — Domain randomization (foot friction, base COM,
encoder bias) is included but minimal. Real deployment may need
additional DR (body mass, damping, motor strength) and the two-stage
training approach from HumanUP (discovery → refinement with strong
regularization).

### Future work

1. **Standing-probability curriculum** — Mix standing + fallen initial
   states, cosine-anneal the standing probability (HoST/HumanUP).
2. **Strong-to-weak torque curriculum** — Start with full torque,
   gradually reduce (Learning to Get Up).
3. **Two-stage training** — Discovery (sparse rewards, weak regu) →
   refinement (dense tracking, strong regu) for sim-to-real.
4. **Multi-pose curriculum** — Train on easy poses (seated) first,
   progress to hard poses (supine/prone).
5. **Action history** — Add 6-10 step action history if partial
   observability is an issue.

## Usage

```bash
# List all registered tasks (Walka-GetUp should appear).
uv run python scripts/list_envs.py

# Train the get-up policy.
uv run python scripts/train.py Walka-GetUp --env.scene.num-envs=4096

# Play back a trained checkpoint in the viewer.
uv run python scripts/play.py Walka-GetUp --checkpoint-file logs/rsl_rl/walka_get_up/DATE/model_N.pt
```

Training is expected to take longer than the velocity task (~2h on RTX
4090 for velocity) due to the harder exploration problem. With
`num_envs=4096` and `max_iterations=15001`, expect ~3-4h on an RTX 4090.