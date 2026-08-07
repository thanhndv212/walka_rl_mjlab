# Get-Up Task Design for Walka

What the `Walka-GetUp` task does, why each reward term is there, and how
the design follows from research on humanoid fall recovery (HumanUP,
HoST, Learning to Get Up, UHG). Most terms are custom
(`src/tasks/get_up/mdp/`); the stock mjlab terms used are noted inline.

**Status (v1.7):** v1.4 (W&B `c3b1ytc6`, `model_9999.pt`) is the best
checkpoint this task has produced — **100% genuine recovery** from
genuinely fallen poses (`scripts/eval_fallen_recovery.py`) — but its
standing pose is wrong: a bent-forward waist, propped by a hand left on
the ground. v1.5 and v1.6 both tried to fix that posture indirectly
(penalize the hand, then fix an exploit that penalty created) and are
reverted: v1.5 collapsed to ~0% recovery, and v1.6 — despite fixing its
specific exploit — regressed all the way to **1.7% genuine recovery**,
statistically indistinguishable from a policy that does nothing (verified
independently this session; its own training-log metric alone had
understated how bad this was). v1.7 reverts to the v1.4 baseline and
reinforces upper-body verticality directly instead. See "Version history
and training campaign log" below for the full log, numbers, and how each
version was verified.

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

**This section describes the v1.7 reward set** (v1.4's task/style/
progress terms, plus v1.7's `upper_body_upright`; v1.5/v1.6 are reverted).
See "Version history and training campaign log" below for how the design
got here — the v1.1 stack this section originally documented
(`base_height_exp` + stock `upright` + `body_up_exp`, all additive) was
replaced wholesale in v1.2 and no longer exists in code.

What each of the 16 reward terms in `RewardManager` does, why it's
there, and how it shapes the get-up motion. The reward structure
follows the composite design from HumanUP (RSS 2025) and HoST (RSS 2025
Best Systems Paper Finalist): a set of task rewards provides the
primary "get up" signal, style rewards (some conditional, some gated to
a rising or standing phase) shape *how* it gets there, and
regularization penalties keep the motion smooth and safe.

| Term | Weight | Kind | One-line purpose |
|---|---|---|---|
| `task_progress` | 6.0 | bonus | Multiplicative height × orientation (pelvis) — the core anti-exploit task signal |
| `stand_on_feet` | 2.5 | bonus | Binary success: both feet contact + height |
| `upper_body_upright` | 3.0 | bonus (v1.7) | Thorax held vertical, gated on `stand_on_feet`'s condition |
| `shank_vertical` | 2.0 | bonus | Shins vertical — stable planted-feet crouch, gated to rising phase |
| `feet_level` | 2.5 | bonus | Both feet at the same height, gated at `SUCCESS_HEIGHT` |
| `height_progress` | 2.0 | bonus | Dense reward for upward pelvis motion since the previous step |
| `feet_force_progress` | 1.0 | bonus | Dense reward for increasing vertical ground-reaction force |
| `hand_supported_rise` | 2.0 | bonus | Reward raising the pelvis while both hands are planted (push-up phase) |
| `foot_advance` | 1.5 | bonus | Reward a foot stepping forward of the pelvis while a hand is still down |
| `dof_pos_limits` | -1.0 | penalty | Stay inside soft joint limits |
| `action_rate_l2` | -0.1 | penalty | Smooth actions (discourage jitter) |
| `self_collisions` | -1.0 | penalty | Avoid self-contact above 10N |
| `joint_vel_penalty` | -0.0001 | penalty | Energy efficiency (low joint velocities) |
| `torques_penalty` | -1e-6 | penalty | Energy (low torques) |
| `stand_still_pose` | -0.5 | conditional | Hold default pose when near standing |
| `termination` | -500.0 | penalty | Strong penalty for terminating/falling |

### Task rewards — the actual objective

**`task_progress`** (`src/tasks/get_up/mdp/rewards.py::task_progress`,
weight 6.0, `height_target=0.75×0.832≈0.62m`, `orientation_threshold=0.99`,
`asset_cfg.body_names=("pelvis",)`) is the primary task signal, added in
v1.2 to replace the v1.1 additive stack (`base_height_exp` + `upright` +
`body_up_exp`) named above. It's a *multiplicative* height × orientation
tolerance — both `height_component` and `orientation_component` must be
satisfied together, continuously, since either near zero collapses the
whole product. Ported from HoST's actual mechanism (verified against
InternRobotics/HoST's public code, which cites "follow 'Learning to Get
Up'"). This is a structurally stronger anti-exploit than the v1.1 additive
terms or height-gating alone: the v1.1 stack let a policy max out
orientation (hold the pelvis vertical from a low kneel) largely
independent of height — see "Confirmed: the kneeling trap" below for the
empirical burst test that caught this.

**`stand_on_feet`** (`src/tasks/get_up/mdp/rewards.py::stand_on_feet`,
weight 2.5, `target_height=SUCCESS_HEIGHT≈0.58m`) is the binary success
signal: returns 1.0 when both feet are in contact with the ground AND the
pelvis is above target height, else 0.0. This prevents reward hacking
where the robot achieves height without actually standing on its feet
(e.g. propping on knees or hands). Mirrors HumanUP's `stand_on_feet` term.

**`upper_body_upright`** (`src/tasks/get_up/mdp/rewards.py::upper_body_upright`,
weight 3.0, `std=√0.2`, `body_name="thorax"`, added v1.7) rewards the
*thorax* body's own verticality — same projected-gravity math as the
stock `upright` class, but targeted at the thorax instead of the pelvis,
and gated on `stand_on_feet`'s exact condition (both feet contact + height
above `SUCCESS_HEIGHT`) rather than always-on. Added because v1.4 stood up
reliably but stayed bent forward at the waist: `task_progress`'s
orientation component reads the *pelvis*'s orientation, and the pelvis can
be near-vertical while `pitch_waist_joint` folds the thorax forward above
it — a blind spot no other term in the v1.4 stack covered. v1.5/v1.6
(reverted — see version history below) tried to fix this indirectly by
penalizing the hand propping up the bend instead of rewarding the posture
directly; `upper_body_upright` is the direct fix. Gating it on the
standing condition (not just height) keeps it from fighting the rising
motion, where the torso legitimately needs to pass through bent
intermediate poses (e.g. `hand_supported_rise`'s push-up phase) — same
conditional-style principle `stand_still_pose` already uses below.

**`shank_vertical`** and **`feet_level`**
(`src/tasks/get_up/mdp/rewards.py`, weights 2.0 / 2.5, added v1.2) are
HoST's `style_shank_orientation` / `style_ground_parallel`: reward a
stable, feet-planted crouch as the intermediate pose to rise from, gated
to the rising phase (`shank_vertical`) or to `SUCCESS_HEIGHT`
(`feet_level`, gated higher than `shank_vertical` so it doesn't fight the
v1.3 asymmetric lunge step described below).

**`height_progress`** and **`feet_force_progress`** (weights 2.0 / 1.0,
added v1.2 as "Step 1" of the research-backed plan below) are HumanUP's
`r_Δheight` / `r_Δfeet_contact_forces`: dense rewards for the pelvis
rising and ground-reaction force increasing step-to-step, giving gradient
far from the target where `task_progress`'s saturating tolerance shape is
~flat — exactly the dead zone the kneeling trap exploited.

**`hand_supported_rise`** and **`foot_advance`** (weights 2.0 / 1.5, added
v1.3) reward a human-like hands-then-lunge get-up strategy observed in
v1.2 rollouts (which converged to face-down, arms/legs spread, and
stalled): push the torso up while both hands are planted
(`hand_supported_rise`), then step one foot forward while a hand is still
down to pivot upright (`foot_advance`).

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

## Version history and training campaign log

Every reward change to this task gets its own commit (`v1.N: ...`) so a
checkpoint is always traceable to the exact reward set that produced it.
This table is the running record of what each version changed and how it
actually performed — kept up to date going forward so a future tuning pass
starts from here instead of re-discovering the same failure modes (the
same discipline `soarm_mjlab/docs/reach_training_debug_log.md` follows for
that package's Reach task).

| Version | Change | Outcome |
|---|---|---|
| v1.1 | Additive `base_height_exp` + stock `upright` + `body_up_exp` | **Kneeling trap.** Reward climbed then flatlined by 4% of the training budget; `upright` pinned near ceiling while genuine fallen-recovery stayed at 0%. See "Confirmed: the kneeling trap" below. |
| v1.1 (fix attempt) | Standing-probability curriculum (mix standing + fallen resets) | **Insufficient alone.** Logged metrics improved, but replaying a checkpoint from a *forced-fallen* reset (curriculum bypassed) showed zero recovery — the metrics tracked curriculum-assisted resets, not a growing recovery skill. |
| v1.2 | Replaced the v1.1 stack with `task_progress` (multiplicative height×orientation, HoST-verified) + `shank_vertical`/`feet_level` (stable-base style) + `height_progress`/`feet_force_progress` (HumanUP delta-progress, "Step 1"/"Step 2" below) | Fixed the kneeling trap structurally (multiplicative reward can't be maxed on orientation alone). Rollouts converged to a *different* stuck point: face-down, arms/legs spread. |
| v1.3 | `hand_supported_rise` + `foot_advance` — reward a human-like hands-then-lunge strategy (push up on both hands, step one foot forward, pivot upright) | Unstuck v1.2's face-down convergence; policy started reaching standing height via a recognizable get-up motion. |
| v1.4 | "Step 3" assistive decaying lift force (`mdp/events.py::assistive_lift_force`) — HoST-style vertical pelvis force while near-vertical, annealing to 0 by step 300k | **10,000-iteration run** (W&B `c3b1ytc6`): `standing_success` climbed to a strong finish, 0.82 at the final logged step (noisy mid-run, dipping to ~0.03 around 75% through, but recovering). **Independently re-verified** against `model_9999.pt` on `scripts/eval_fallen_recovery.py`'s dedicated (curriculum-bypassed) harness: **100% genuine recovery (60/60)**, peak height mean 0.805m, **+0.295m over a matched zero-action baseline** (0.510m) — a real, large skill signal, not reset-physics noise. Video-confirmed (genuinely-fallen reset, `--no-terminations`, 400 steps): reliably reaches a wide, stable two-foot stance by ~2s in and holds it — but the torso is visibly hunched/bent forward with an arm raised, not a straight standing pose, matching the "bent forward at the waist" finding. This is the checkpoint v1.7 is built on top of. |
| v1.5 *(reverted)* | `hands_off_ground` penalty (weight -3.0) for hand contact once standing; capped `foot_advance` at `SUCCESS_HEIGHT` | **Collapsed.** 15,000-iteration run (W&B `1q3ugie0`): `standing_success` fell to **0.0006 at the final step** — essentially zero — while `Episode_Reward/foot_advance` climbed toward its ceiling throughout, a camping exploit the height cap itself created (park the foot forward, keep a hand down, stay just below `SUCCESS_HEIGHT`, collect free reward forever; crossing the gate now cost guaranteed income it didn't cost before). |
| v1.6 *(reverted)* | Rewrote `foot_advance` as a delta-progress term (rewards the *increase* in forward offset, not the maintained condition) — closes the v1.5 camping exploit structurally | Closed the exploit (CPU-verified: cumulative zero-action reward bounded at ~0.2–1.5 over 300 steps, vs. v1.5's unbounded ~300) but **did not fix the underlying task**. Trained to a full **15,000-iteration campaign** (W&B `jgn5np4w`, preceded by a 1,500-iteration burst check, `b7eiotwx`, `standing_success≈0.18`, that looked promising enough to justify the full run) — training-log `standing_success` peaked early at 0.54 (iteration ~2,645, ~18% through), degraded through the middle (~0.04–0.06 for iterations 6,000–12,000), bottomed near-zero around iteration 13,500, and only partially recovered to 0.16 by the end. **Independently re-verified** against `model_14999.pt` on `scripts/eval_fallen_recovery.py`: **1.7% genuine recovery (1/60)**, and a **+0.019m peak-height delta over the zero-action baseline — within noise, i.e. statistically indistinguishable from doing nothing.** Video-confirmed (same genuinely-fallen-reset setup as v1.4 above): across the full 8s rollout the robot stays sprawled flat, limbs splayed outward, never rising — this is not a posture problem, it essentially never gets up from a genuine fall at all. The 0.16 end-of-training W&B metric — averaged over a mix of curriculum-assisted and genuinely-fallen resets during training — was actively misleading on its own, exactly the trap "Validation principle: never trust logged episode averages alone" (below) warns about. |
| **v1.7** *(current)* | **Reverted v1.5/v1.6 back to v1.4**, added `upper_body_upright` (weight 3.0) — rewards the thorax's own verticality directly, gated on `stand_on_feet`'s condition | Addresses the v1.4 finding at its source instead of through the hand-contact proxy v1.5/v1.6 used. CPU-verified: fires (~0.93–0.997) once both feet are planted and the pelvis is at standing height, exactly zero before that gate opens. **Not yet training-verified** — next step is the tiered validation strategy below (CPU check ✅ done → GPU burst → full campaign) before trusting it beyond the CPU check. |

Takeaway pattern across v1.4→v1.6: two consecutive versions (v1.5, v1.6)
fixed real problems they found (an unpenalized hand, then an exploit their
own fix created) without ever fixing the problem that started the
investigation — and each fix cost real training stability, not just wasted
effort: v1.5 collapsed `standing_success` to zero, and v1.6, despite
closing v1.5's specific exploit, regressed all the way from v1.4's 100%
genuine recovery to 1.7% — indistinguishable from a policy that does
nothing. Its own training-log metric (0.16 at the final iteration) actively
understated how bad this was, since that metric is contaminated by
curriculum-assisted resets; the dedicated eval harness and direct video
inspection are what caught it. Neither v1.5 nor v1.6 added a reward for the
actual target behavior (a straight upper body) — only penalties/patches
around its absence, and in v1.6's case a patch that came at a real cost to
the underlying recovery skill. v1.7 breaks that pattern by reinforcing the
goal state directly, on top of the v1.4 checkpoint that is — as of this
verification — still the best one this task has produced.

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

### Ground-clearance correction (v1.1)

The `z` and roll/pitch ranges above are sampled independently: roughly a
`(60°/360°)²` slice of the roll×pitch space lands near upright while `z`
is still drawn from the near-ground range meant for lying-flat poses. For
an upright orientation, standing height (0.832m) is set almost entirely by
leg extension, so a low `z` there drives the legs tens of centimeters into
the ground plane. Confirmed empirically before this fix: a zero-action
rollout showed 7/64 fresh resets launching past the `too_high` (1.2m)
termination bound within 15 steps — the contact solver resolving that
interpenetration as one explosive correction, a simulator artifact rather
than a get-up behavior.

`ensure_ground_clearance` (`mdp/events.py`, last "reset" event) fixes this
by construction rather than by re-tuning the ranges (which can't rule out
the upright/low-`z` combination without losing near-ground coverage for
genuinely fallen poses). It computes each robot geom's world-space AABB
(`mj_model.geom_aabb` rotated by `geom_xmat`) and shifts the root strictly
upward so nothing penetrates the ground plane, plus a small clearance
margin. It's a conservative bound (AABB, not exact mesh), so it
occasionally over-corrects a valid pose by a few centimeters — benign,
since the robot just free-falls that small gap under gravity in the first
physics step or two, similar in spirit to the literature's "drop from a
small height and settle" approach (see Research-backed implementation
plan).

### Standing-probability curriculum (v1.1) — insufficient alone

The original design intent (see "Reconciled Design Decisions" in the
internal research notes) called for mixing standing + fallen resets from
the start, not sampling purely from the fallen distribution — this was
deferred during implementation (`cfg.curriculum` was left empty in v1).
`reset_to_standing_curriculum` (`mdp/events.py`) adds it back: a
Bernoulli-annealed fraction of resets (`start_prob=0.5 → end_prob=0.05`
over the first 150,000 env-steps, tracked via `env.common_step_counter`)
are overwritten to the default standing pose plus small joint noise,
instead of the fallen distribution.

**This alone did not fix the kneeling trap** — see "Confirmed: the
kneeling trap" under Known Risks for the full empirical account. In short:
logged episode metrics improved substantially early on, but a checkpoint
pulled at that "peak" and replayed from a genuinely fallen start (curriculum
bypassed) showed zero recovery — the metrics were tracking the shrinking
fraction of curriculum-assisted free successes, not a growing recovery
skill.

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

- **[HumanUP](https://arxiv.org/abs/2502.12152)** (RSS 2025) — Two-stage
  RL (discovery → refinement) for Unitree G1 get-up. Task reward is an
  *additive* sum including explicit height/contact-force *progress*
  terms (`r_Δheight`, `r_Δfeet_contact_forces`), not the multiplicative
  height×upright×feet combination this doc originally assumed (corrected
  after re-checking the paper — see Known Risks). Real-world deployment,
  78.3% vs. 41.7% success vs. the G1's stock controller.
- **[HoST](https://arxiv.org/abs/2502.08378)** (RSS 2025 Best Systems
  Paper Finalist) — Multi-critic reward groups (task/style/regularization/
  post-task, each with its own value function, combined via normalized
  advantages), an **assistive decaying lift force** during training (not
  a standing-probability curriculum — corrected attribution), and action
  rescaling for sim-to-real. Deployed on Unitree G1.
- **Learning to Get Up** (SIGGRAPH 2022) — Three-stage curriculum
  (discover → weaker → slower). Key finding: direct regularization
  without curriculum → FAILS. Strong-to-weak torque curriculum
  essential.
- **[FRASA](https://arxiv.org/html/2410.08655)** — CrossQ-based, trains
  in 13-37 minutes. Uses exactly this task's standing-probability idea
  ("state occasionally initialized close to neutral pose... providing
  occasional large rewards"), without the height-progress or reward-
  gating mechanisms above — useful as confirmation the curriculum idea
  isn't wrong, just insufficient alone.
- **[HiFAR](https://arxiv.org/abs/2502.20061)** (Feb 2025) — Multi-stage
  curriculum specifically for high-dynamics fall recovery, progressively
  incorporating more complex/high-dimensional recovery tasks.
- **[Learning to Get Up Across Morphologies](https://arxiv.org/pdf/2512.12230)**
  (Dec 2025) — Zero-shot get-up recovery across a family of morphologies
  with a single unified policy.
- **[Stubborn](https://arxiv.org/pdf/2606.12814)** (2026) — Unified RL
  framework combining motion tracking and fall recovery in one policy.
- **PHC** (ICCV 2023) — Fail-state recovery, progressive primitives.

### Key insights applied

1. **Conditional style penalties** — `stand_still_pose` is zeroed during
   the rising phase (scaled by proximity to standing height). From
   HumanUP/HoST.
2. **Height + orientation** — `base_height` alone risks the kneeling
   trap; `upright` + `body_up` ensure the torso is actually vertical.
   From Learning to Get Up ablations. (In practice this wasn't enough —
   see Known Risks: `upright`/`body_up` turned out to be satisfiable
   *without* height, which is exactly the trap that occurred.)
3. **Binary success signal** — `stand_on_feet` prevents reward hacking
   where height is achieved without standing on feet. From HumanUP.
4. **Strong termination penalty** — -500 weight discourages early
   termination. From HumanUP/HoST.
5. **Standing-probability curriculum (v1.1)** — Added after v1's
   single-distribution training hit the kneeling trap in practice, not
   just in theory. Confirmed insufficient alone; see "Research-backed
   implementation plan" in Known Risks for what the literature suggests
   next (progress rewards, height-staged gating, assistive lift force).

## Known risks and future work

### Confirmed: the kneeling trap (v1.1, empirical)

The kneeling trap this section originally flagged as a theoretical risk
happened, was diagnosed, and a first mitigation (the standing-probability
curriculum described above) was tried and found insufficient on its own.
What's now confirmed from a full `num_envs=4096` run on a rented RTX 4090:

| Iteration | Anneal progress | `standing_success` | `base_height` reward | `stand_on_feet` reward |
|---|---|---|---|---|
| 600-700 (peak) | ~10% through anneal | 0.43 | 2.3 / 5.0 | 1.08 / 2.5 |
| 5200 | ~84% through anneal | 0.11 | 0.60 | 0.27 |
| 6423 (anneal saturated at floor) | 100% | 0.086 | 0.48 | 0.21 |

- The trap is real and reproducible: reward climbed rapidly then flatlined
  within ~600/15001 iterations (4% of the budget), with `upright` pinned
  near its 1.0 ceiling (0.97+) for the remaining 96% while `base_height`/
  `stand_on_feet` stayed near zero.
- Mixing in curriculum-standing resets raised the *logged* metrics (some
  fraction of episodes start already succeeding) without changing the
  underlying policy's ability to recover from an actually-fallen state.
  Replaying the iteration-700 "peak" checkpoint from a forced-fallen reset
  (curriculum bypassed) showed **zero recovery** — pelvis height converged
  to ~0.07m and held a static "torso up, one leg splayed flat on the
  ground" pose for the full rollout, reproduced across multiple seeds and
  at a much later checkpoint (iteration 6900) alike.
- Root cause is the reward landscape, not the initial-state distribution:
  `base_height_exp`'s narrow Gaussian (`std=0.1`) around the 0.832m target
  gives essentially no gradient anywhere far from standing height, while
  `upright`/`body_up` are large, easy, and height-independent. Changing
  *where* episodes start doesn't change *what gets rewarded once fallen* —
  the policy still has no local incentive to move up from the ground.

### Research-backed implementation plan

A literature pass across HumanUP (RSS 2025), HoST (RSS 2025 Best Systems
Paper Finalist), FRASA, and HiFAR — specifically hunting for how each
avoids this exact failure mode — turned up concrete mechanisms this task's
reward/curriculum don't have yet. Ranked by expected impact for our
specific failure (cheapest/most targeted first):

1. **Height/contact-force *progress* reward** (HumanUP's `r_Δheight`,
   `r_Δfeet_contact_forces`) — **highest priority, `mdp/rewards.py`-only
   change.** HumanUP's task reward is an additive sum
   (`r_up = r_height + r_Δheight + r_uprightness + r_stand_on_feet +
   r_Δfeet_contact_forces + r_symmetry`) that, critically, rewards the
   pelvis height *increasing* step-to-step and the feet contact force
   *increasing* step-to-step — not just "are you at the target." (Note:
   this doc's "Reconciled Design Decisions" notes assumed HumanUP used a
   *multiplicative* height×upright×feet combination; re-checking the
   paper text, it's additive with explicit progress terms — correcting
   that assumption here.) A progress term gives dense gradient from the
   very first fallen frame, independent of how far the pelvis is from
   0.832m, directly patching the "zero gradient far from target" dead zone
   that let `upright` win by default. Implementable as
   `clip(h_t - h_{t-1}, min=0)` (only reward upward motion, not downward,
   to avoid rewarding controlled collapse) plus the equivalent for
   `feet_ground_contact` force magnitude. Needs one extra piece of state
   (previous height) tracked across steps — a class-based reward term with
   a `reset()` hook, the same pattern `mdp.last_action` uses for
   `action_rate_l2`.

2. **Height-staged reward gating** (HoST) — **second priority,
   `mdp/rewards.py`-only change.** HoST explicitly gates each reward group
   behind a height stage: an `upright`/`body_up`-equivalent term doesn't
   pay out until the robot has first crossed a minimum pelvis height. This
   directly closes the exploit filmed above — "hold torso vertical while
   sitting on the ground" stops being free once `upright` requires
   `h > ~0.3-0.4m` first. Cheapest version: multiply `upright`/`body_up`'s
   existing reward functions by a `(h > stage_threshold)` gate, mirroring
   how `stand_still_pose` is already conditionally scaled by height
   proximity — same pattern, opposite direction (gate low-height instead
   of high-height).

3. **Assistive decaying lift force** (HoST) — **most direct fix for the
   ground→standing transition, moderate lift.** HoST applies an upward
   external force on the robot's base once the trunk is near-vertical,
   fading as the robot learns to hold height unassisted. Unlike the
   standing-probability curriculum (which only changes the *initial
   state*), this intervenes *during* the fallen episode itself, so the
   value function experiences near-target states derived from actually
   being fallen and pushed up — teaching the lift, not just the hold.
   Implementable as a "step"-mode `EventTermCfg` calling
   `write_external_wrench_to_sim` on the pelvis body, gated on
   `projected_gravity_b[:, 2] < -threshold` (trunk near-vertical) and
   annealed down via `env.common_step_counter`, the same pattern
   `reset_to_standing_curriculum` already uses.

4. **Multi-critic reward architecture** (HoST) — **biggest lift, touches
   the RL algorithm, not just the env config.** HoST splits the reward
   into 4 groups (task / style / regularization / post-task), each with
   its own value function, combined via *normalized* advantages rather
   than summed raw values — structurally preventing one term (`upright`)
   from dominating gradient signal just because it's easy, regardless of
   its raw weight relative to `base_height`'s. RSL-RL's stock
   `PpoAlgorithm`/`OnPolicyRunner` assume a single scalar reward and
   critic; this would need either a custom multi-critic runner (mirroring
   `GetUpOnPolicyRunner`'s existing customization point in `rl/runner.py`)
   or grouping rewards into 2-3 `RewardManager`-level buckets with
   separately-normalized running statistics before summing. Sequence this
   after 1-3 — those are cheap enough to test in isolation first.

5. **Two-stage discover → refine training** (HumanUP) — **structural
   process change, not a single env tweak.** Train Stage I with weak
   regularization (drop `action_rate_l2`, `joint_vel_penalty`,
   `torques_penalty` weights toward 0) purely to discover *any* working
   get-up trajectory; then Stage II imitates/tracks a slowed-down version
   of what Stage I found, under the full regularization this doc's reward
   table already specifies. HumanUP's Stage II also generates its fallen-
   pose pool this way (see item 6) rather than sampling live. Most proven
   fix in the literature, most engineering work: needs a reference-
   trajectory extraction step and a tracking reward, likely as a second
   task variant (e.g. `Walka-GetUp-Refine`) consuming Stage I's output.

6. **Pre-simulated drop-and-settle pose pool** (HumanUP) — replaces the
   live analytic `ensure_ground_clearance` correction with the
   literature-standard approach: pre-generate a large pool of fallen poses
   offline by randomizing joints from a canonical lying pose, dropping
   from ~0.5m, and simulating ~10s to let self-collisions resolve
   naturally, then sample from that pool at reset instead of computing a
   geometric correction live. More physically faithful, but a bigger
   one-time engineering cost than the AABB nudge currently in place —
   worth revisiting once the reward-side fixes above are validated, not
   before.

See "Implementation and validation roadmap" below for the step-by-step
plan to build and validate items 1-6 without repeating the mistake that
caused this section to exist: trusting logged episode averages that
turned out to be inflated by the curriculum itself.

### Other known risks

**No action history** — v1 relies on RSL-RL's implicit history handling.
If partial observability hurts, explicit action history (6-10 steps, as
in HoST/HumanUP) can be added.

**Sim-to-real gap** — Domain randomization (foot friction, base COM,
encoder bias) is included but minimal. Real deployment may need
additional DR (body mass, damping, motor strength) and the two-stage
training approach from HumanUP (discovery → refinement with strong
regularization) — see item 5 above.

## Implementation and validation roadmap

Concrete steps for building and validating the six research-backed items
above, in the order they should be attempted. Each step has an
implementation sketch, a validation gate, and an explicit cost estimate —
the goal is to catch a non-working idea in minutes of CPU time or a few
dollars of GPU time, not after a full multi-hour rented run, which is
exactly how the standing-probability curriculum's failure was discovered
the expensive way.

**Status: Steps 1–3 implemented** (v1.2: Steps 1 & 2, as `height_progress`/
`feet_force_progress` and the `shank_vertical`/`feet_level`/`foot_advance`
height gates; v1.4: Step 3, as `assistive_lift_force`) — see "Version
history and training campaign log" above for how each performed. Steps 4–6
remain open; none have been attempted. The bent-waist problem v1.4 exposed
turned out not to need any of Steps 4-6 — it needed a direct upper-body
reward, added in v1.7 (see version history), a different kind of gap than
this roadmap's items were scoped for (which target *reaching* standing
height at all, not standing posture once there).

### Validation principle: never trust logged episode averages alone

The standing-probability curriculum's logged `standing_success`/
`base_height` looked great at iteration ~700 and only revealed itself as
hollow when a checkpoint was replayed from a **forced-fallen** reset,
bypassing the curriculum. Every step below must be checked the same way
before it's trusted. Concretely:

- **Step 0 (do this first, ~30 min)**: turn the ad-hoc inspection script
  used to diagnose the trap into a small permanent one,
  `scripts/eval_fallen_recovery.py`. It should: build the env with
  `reset_to_standing_curriculum` removed from `cfg.events` (or
  `env.common_step_counter` forced high, which drives the curriculum to
  its floor — removing the event outright is more robust since it doesn't
  depend on the anneal schedule staying at its current values), load a
  checkpoint via the same `runner_cls(...).load(...)` /
  `get_inference_policy()` pattern `scripts/play.py` uses, roll out N
  episodes (e.g. 20 seeds × 500 steps), and report: fraction of episodes
  reaching `stand_on_feet`'s condition (both feet contact + `h > 0.7`)
  for at least 1 continuous second, and peak pelvis height reached per
  episode. This is the one number to trust from here on — not the
  training log's `Episode_Metrics/standing_success`, which is an average
  over whatever mix of curriculum-assisted and genuinely-fallen episodes
  happened to complete in a given iteration's rollout window.
- **Tiered compute for every subsequent step**: (a) a CPU correctness
  check — few hundred iterations, `num_envs=8-64`, just confirming no
  crash/NaN and that new reward terms produce sane, non-degenerate values
  (as done for the ground-clearance fix); (b) a short, cheap GPU burst —
  `num_envs=512-1024`, ~1000-2000 iterations, ~10-15 minutes, checked
  against the Step 0 harness for an early trend, not full convergence;
  (c) only once (b) shows the eval-harness numbers trending up does a full
  `num_envs=4096`, `max_iterations=15001` rented-GPU run get justified.

### Step 1 — Height & contact-force progress reward — ✅ Done (v1.2)

**Highest priority.** Add to `src/tasks/get_up/mdp/rewards.py` as a
class-based term (the officially supported stateful pattern — see
`mjlab.managers.manager_base.ManagerTermBase`/`ManagerTermBaseCfg`'s
docstring: a class auto-instantiated with `(cfg, env)`, called each step,
with an optional `reset(env_ids)`):

```python
class height_progress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self._prev_h = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids):
        self._has_prev[env_ids] = False  # don't read root_link_pos_w here — see note below

    def __call__(self, env, asset_cfg=_DEFAULT_ASSET_CFG):
        h = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
        delta = torch.where(self._has_prev, (h - self._prev_h).clamp(min=0.0), torch.zeros_like(h))
        self._prev_h, self._has_prev = h.clone(), torch.ones_like(self._has_prev)
        return delta
```

**Do not** read `root_link_pos_w` inside `reset()` to seed `_prev_h` —
`RewardManager.reset()` runs inside `_reset_idx()`, *before*
`sim.forward()` refreshes kinematics from the just-applied reset events
(the exact staleness trap `ensure_ground_clearance` already had to work
around; see its docstring). The `_has_prev` flag sidesteps this by
deferring the baseline capture to the first real `__call__`, which only
happens after the env's own `sim.forward()` inside `step()`/`reset()` has
already run.

`clamp(min=0.0)` rewards only upward motion — a policy that controls its
descent (e.g. a soft landing) shouldn't be penalized, but shouldn't be
paid for falling either. Mirror this for feet contact force using
`feet_ground_contact`'s force field (same pattern, previous-force state
instead of previous-height). Start both at a modest weight (e.g. 1.0-2.0,
comparable to `stand_on_feet`'s 2.5) rather than guessing HumanUP's exact
coefficients — this reward's job is to add *gradient*, not to dominate
the total.

**Validation gate**: CPU correctness check first — write a tiny script
that manually raises the robot's `qpos` z between two `env.step()` calls
(or just watch the reward value during a zero-action rollout from a
naturally-settling fallen pose) and confirm the reward is positive while
height increases, zero while it's flat or decreasing. Then the tiered
compute strategy above, gated on Step 0's harness.

### Step 2 — Height-staged reward gating — ✅ Done (v1.2)

**Second priority**, same file. `upright` is a stock **class-based** term
(`mjlab.tasks.velocity.mdp.rewards.upright`, auto-instantiated by
`RewardManager` — not a plain function), so it can't be wrapped by simply
calling it like a function. Cheapest correct approach: re-derive the same
few lines of math locally rather than reuse the stock class. Its core is:

```python
projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
# upright reward is exp(-xy_squared / std**2); gate it:
gate = (h > stage_threshold).float()
return torch.exp(-xy_squared / std**2) * gate
```

`body_up_exp` is already ours (a plain function in this task's
`rewards.py`) — gating it is a one-line change: multiply its existing
`clamp(-pg[:, 2], 0, 1)` by `(h > stage_threshold).float()`. Pick
`stage_threshold` low enough to still reward genuine early progress (e.g.
~0.3-0.4m, well below the 0.7m `stand_on_feet` threshold) — the goal is
closing the "free reward while flat on the ground" exploit, not making
the reward sparse again.

**Validation gate**: CPU check that the gated reward is exactly 0 for a
robot held at `h < stage_threshold` (e.g. the exact static pose filmed in
the kneeling-trap video) and behaves identically to the ungated version
above threshold. Then combine with Step 1 and run the same tiered compute
strategy — this is the first point where Step 0's harness should show a
real signal difference from the original run.

### Step 3 — Assistive decaying lift force — ✅ Done (v1.4)

**Most direct fix for the ground→standing transition, moderate lift.**
New "step"-mode `EventTermCfg` (runs every `env.step()`, not just on
reset) calling `Entity.write_external_wrench_to_sim(forces, torques,
env_ids, body_ids)` on the pelvis body — forces/torques are world-frame
`(N, num_bodies, 3)` tensors, and **persist until the next call or a
reset**, so the force must be explicitly re-applied (or zeroed) every
step, not just when the gating condition first becomes true:

```python
def assistive_lift_force(env, max_force, near_vertical_threshold, anneal_steps, asset_cfg=_DEFAULT_ASSET_CFG):
    asset = env.scene[asset_cfg.name]
    pg_z = asset.data.projected_gravity_b[:, 2]
    near_vertical = (pg_z < -near_vertical_threshold).float()
    decay = 1.0 - min(1.0, env.common_step_counter / anneal_steps)
    force_z = max_force * decay * near_vertical
    forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    forces[:, 0, 2] = force_z
    torques = torch.zeros_like(forces)
    asset.write_external_wrench_to_sim(forces, torques, body_ids=[pelvis_body_id])
```

registered as `EventTermCfg(func=assistive_lift_force, mode="step",
params={...})`. `max_force` needs empirical tuning — too strong and the
policy never learns to lift itself (permanent training wheels); too weak
and it doesn't bridge the gap that caused the trap. Start conservative
(a fraction of body weight) and check via the CPU correctness pass
whether it measurably slows/reverses descent for a robot that would
otherwise settle to the ground, without fully carrying it to standing
height on its own.

**Validation gate**: same tiered strategy. This is the step most likely
to need a few retuning passes at the cheap-GPU-burst tier before a full
run is justified — track how the Step 0 harness's peak-height distribution
shifts as `max_force` and `anneal_steps` change.

### Step 4 — Multi-critic reward architecture — not attempted

**Biggest lift — confirmed to need real engineering, not a config
change.** Checked directly: `rsl_rl.algorithms.ppo.PPO` (the algorithm
`MjlabOnPolicyRunner`/`GetUpOnPolicyRunner` build on) hard-codes a single
`self.critic` producing one scalar value and one value loss — there's no
built-in support for HoST's per-group critics + normalized-advantage
combination. This means: a custom `PPO` subclass holding a
dict/list of critics (one per reward group), rollout storage extended to
keep per-group rewards/values/advantages, and an `update()` that
normalizes each group's advantages before summing — wired in via a
custom runner alongside `GetUpOnPolicyRunner`'s existing customization
point in `rl/runner.py`. Only worth this investment if Steps 1-3 combined
still don't clear Step 0's harness at a reasonable rate — prototype the
normalized-advantage-combination logic against synthetic reward groups
first (a plain unit test, no simulation needed) before wiring it into the
full training loop, since debugging it live against a 4096-env rollout is
far more expensive than debugging it against fixed synthetic data.

### Step 5 — Two-stage discover → refine training — not attempted

**Structural process change, most proven fix in the literature, most
work.** Only pursue if Steps 1-4 don't produce a working single-stage
policy. Plan: a `Walka-GetUp-Discover` variant of this task with
`action_rate_l2`/`joint_vel_penalty`/`torques_penalty` weights pushed
toward 0 (Stage I's job is finding *any* working trajectory, not a
deployable one), trained until Step 0's harness shows consistent
recovery; then extract a reference trajectory (log `qpos`/joint positions
from a successful rollout) and build `Walka-GetUp-Refine` with a
DoF/body tracking reward against that reference (mirroring HumanUP's
`r_tracking_DoF + r_tracking_body`) plus this doc's full regularization
table, training a second policy to imitate a slowed-down version of it.

### Step 6 — Pre-simulated drop-and-settle pose pool — not attempted

**Independent polish, not a fix for the core exploration problem —
sequence in parallel with the above or last.** Replaces the live
analytic `ensure_ground_clearance` correction with the literature-
standard approach: an offline script samples joints from a canonical
lying pose, drops the robot from ~0.5m, and simulates ~10s to let
self-collisions resolve naturally, saving the resulting `qpos` into a
pool; the `reset_base` event then samples from that pool instead of
computing a geometric correction live. More physically faithful (no AABB
over-correction "hover"), but doesn't by itself address why the policy
never learns to stand once fallen — the diagnosis above points squarely
at the reward landscape, not pose realism.

### Budget and stop conditions

At time of writing, ~$2.12 of vast.ai credit remains after this task's
false starts. Reserve full-scale rented-GPU runs for after a cheap GPU
burst (tiered strategy above) shows a clear, Step-0-harness-confirmed
upward trend — not for testing whether an idea works at all. If Steps 1-3
combined don't move Step 0's harness numbers in a short burst, stop and
re-diagnose (checked reward magnitudes, checked the gating threshold,
checked the force curriculum's tuning) before spending on Steps 4-5,
which are substantially more expensive to build and debug.

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

### Storage: W&B is short-term, Hugging Face Hub is long-term

See docs/vast_ai_training.md's "Storage strategy" section for the full
reasoning. In short: W&B's checkpoint uploads (every `save_interval`
iterations) have a hard artifact-storage quota that a handful of
15,000-iteration runs at the old default exhausted outright — `rl_cfg.py`
now saves every 1500 iterations instead of 100. For a checkpoint actually
worth keeping past the run itself, push it to Hugging Face Hub:

```bash
uv run python scripts/push_to_hub.py \
    --repo-id <your-hf-username>/walka-get-up \
    --wandb-run-path <entity>/<project>/<run_id> \
    --experiment-name <the --agent.experiment-name that run used> \
    --task-title "Walka get-up" \
    --task-description "For a Walka biped fall-recovery task: the policy commands Walka to rise from a fallen pose to standing."
```

`--experiment-name` must match the run's actual `--agent.experiment-name`
(this session used distinct per-run names like `walka_get_up_v15_15k`, not
a single shared one) — it's how the script finds the right local
`logs/rsl_rl/<experiment_name>/` staging directory when pulling from W&B.