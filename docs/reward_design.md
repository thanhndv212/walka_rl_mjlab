# Reward design for Walka velocity tasks

What each of the 16 reward terms in `RewardManager` does, why it's there, and
how it shapes the gait seen in the trained `Walka-Flat` policy
(`docs/images/walka_flat_trained.gif`). Most terms come straight from
mjlab's stock velocity template (`mjlab.tasks.velocity.velocity_env_cfg`);
`self_collisions`, `foot_gait`, and `stand_still` are added explicitly in
`src/tasks/velocity/config/walka/env_cfgs.py`. Every reward is computed
every step and summed as `Σ weight_i × term_i(env)` — the numbers below are
per-step contributions, not per-episode totals (`Episode_Reward/*` in W&B
is that sum's mean over an episode).

| Term | Weight | Kind | One-line purpose |
|---|---|---|---|
| `track_linear_velocity` | 2.0 | bonus | Match commanded forward/lateral velocity |
| `track_angular_velocity` | 2.0 | bonus | Match commanded yaw rate |
| `upright` | 1.0 | bonus | Keep the pelvis level |
| `pose` | 1.0 | bonus | Stay near default joint angles, tolerance scales with speed |
| `foot_gait` | 0.5 | bonus | Match a fixed alternating stance/swing clock |
| `dof_pos_limits` | -1.0 | penalty | Stay inside soft joint limits |
| `self_collisions` | -1.0 | penalty | Avoid self-contact above 10N |
| `stand_still` | -1.0 | penalty | Hold the default pose when no command is active |
| `foot_clearance` | -2.0 | penalty | Lift feet to target height during swing |
| `foot_swing_height` | -0.25 | penalty | Hit target peak swing height at landing |
| `action_rate_l2` | -0.1 | penalty | Smooth actions (discourage jitter) |
| `foot_slip` | -0.1 | penalty | No foot sliding while in contact |
| `soft_landing` | -1e-5 | penalty | Soft footfalls (tiny weight — see below) |
| `body_ang_vel` | 0.0 | inert | Wired but never given a nonzero weight — see "Inert terms" |
| `angular_momentum` | 0.0 | inert | Wired but never given a nonzero weight — see "Inert terms" |
| `air_time` | 0.0 | inert | Wired but never given a nonzero weight — see "Inert terms" |

## Command tracking — the actual task

**`track_linear_velocity`** (`mjlab.tasks.velocity.mdp.rewards.track_linear_velocity`,
weight 2.0, `std=√0.25`) and **`track_angular_velocity`** (same file, weight
2.0, `std=√0.5`) are the only terms that reward *doing the task* — matching
the sampled `twist` command's linear (x/y) and angular (yaw) velocity.
Both are `exp(-error² / std²)`, i.e. saturate near 1.0 when tracking is
good and decay smoothly (not a cliff) as error grows. Everything else in
this doc is reward *shaping*: bias PPO toward a gait that looks and moves
like a biped rather than the cheapest way to rack up tracking reward (e.g.
scooting on the ground, or vibrating in place).

The `twist` command itself isn't static: `Curriculum/command_vel` (mjlab's
stock curriculum, unmodified here) widens the sampled velocity range over
training — `lin_vel_x` starts at `(-1, 1)` and reaches `(-2, 3)` by
iteration `10000×24` steps, `ang_vel_z` similarly widens from `±0.5` to
`±0.7`. The final training run's logs confirm it reached the widest stage
(`lin_vel_x_max=3.0`, `ang_vel_z_max=0.7`) by the end of the 10001
iterations. `pose` and `foot_gait` (below) both read this same command to
decide *how much* freedom of motion to allow, so the curriculum affects
more than just what velocities get commanded.

## Balance & posture

**`upright`** (class `upright` in the same file, weight 1.0, `std=√0.2`,
`asset_cfg.body_names=("pelvis",)` set in `env_cfgs.py`) penalizes the
pelvis's tilt relative to world-up (projected gravity's xy component in the
pelvis frame). This is the main thing keeping the robot from falling over
sideways or pitching forward — `Episode_Reward/upright ≈ 0.99` in the
trained run (near the 1.0 ceiling) is a decent proxy for "stayed upright
essentially the whole episode."

**`pose`** (class `variable_posture`, weight 1.0) penalizes deviation from
the default joint pose (`INIT_STATE.joint_pos` in `walka_constants.py`,
`exp(-mean(error² / std²))` per-joint), but the tolerance (`std`) isn't
fixed — it switches between three regimes based on total commanded speed
(`linear + angular`, see the function's `walking_threshold=0.05` /
`running_threshold=1.5` defaults, unmodified from stock): `std_standing`
(tight, `{".*": 0.05}` — hold the default pose precisely when not
commanded to move), `std_walking`, and `std_running` (looser per-joint,
tuned in `env_cfgs.py` — e.g. `pitch_knee_joint` gets `0.35` walking /
`0.6` running vs. `roll_ankle_joint`'s `0.1` / `0.15`, reflecting how much
each joint actually needs to move for a stride vs. how much motion would
break balance). This is what stops PPO from finding a "gait" that tracks
velocity via some joint-limit-abusing shortcut instead of an actual
walking motion.

**`stand_still`** (`src/tasks/velocity/mdp/rewards.py`, weight -1.0,
`command_threshold=0.1`) is the complement of `pose`'s standing regime: it
penalizes joint deviation from default specifically when the *current*
command magnitude is below threshold, gated off entirely once a real
command is active (`scale = (total_command <= command_threshold)`). Two
terms doing similar-sounding things isn't redundant in practice — `pose`
softly biases toward default across all three speed regimes, while
`stand_still` is a harder, walking/running-exempt penalty aimed
specifically at killing residual fidgeting while standing still.

**`body_ang_vel`** (`body_angular_velocity_penalty`, `asset_cfg.body_names
=("pelvis",)` set in `env_cfgs.py`) computes the pelvis's xy angular
velocity penalty correctly and is logged every step
(`Episode_Reward/body_ang_vel` in W&B), but see "Inert terms" below — it
currently contributes nothing to the actual reward signal.

## Gait shaping

**`foot_gait`** (`src/tasks/velocity/mdp/rewards.py`, weight 0.5, `period=
0.6s`, `offset=[0.0, 0.5]`, `threshold=0.56`) is the clock-based gait
reward added on top of stock mjlab (see README's "Gait-phase reward
system" note) — a fixed sine-clock schedule of which foot *should* be in
stance vs. swing (each foot's phase offset by half a cycle from the
other), rewarded 1:1 against measured ground contact
(`feet_ground_contact` sensor). This is what turns "track the velocity
command" into "track it *by alternating footsteps*" instead of, say,
hopping on both feet at once — it's the single biggest lever against a
degenerate non-walking gait. Gated off below `command_threshold=0.1` (no
gait to enforce while standing still), and paired with the `phase`
observation (`src/tasks/velocity/mdp/observations.py`) so the policy
actually has the clock signal available as an input, not just as a
training-time reward.

**`foot_clearance`** (weight -2.0, `target_height=0.1m`, read off
`foot_height_scan`, a `TerrainHeightSensor` wired to `footL`/`footR` sites
in `env_cfgs.py`) penalizes `|height - target| × horizontal_foot_velocity`
— i.e. only while a foot is actually moving horizontally (mid-swing), not
while planted. At -2.0 this has the largest single weight of any penalty
term, which is deliberate: without it PPO tends to find a "shuffling"
gait that barely clears the ground (cheap on `action_rate_l2` and
`foot_slip`, but not a real walk).

**`foot_swing_height`** (weight -0.25) is a complementary, sparser check —
it tracks each foot's *peak* height during a swing phase and penalizes
deviation from `target_height=0.1m` only at the moment of landing
(`compute_first_contact`), rather than continuously like `foot_clearance`.
Continuous + landing-checkpoint together discourage both "never lifts the
foot" and "lifts it but comes down short/long."

**`foot_slip`** (weight -0.1) penalizes horizontal foot velocity *while in
contact* — a planted foot that's still sliding wastes tracking accuracy
and is a real-hardware failure mode (foot slip is loud and reduces
traction), not just a sim artifact.

**`soft_landing`** (weight -1e-5) penalizes contact-force magnitude at the
instant of first contact. The weight is five orders of magnitude smaller
than `foot_clearance` on purpose — raw ground-reaction forces are large
numbers (hundreds of N, see `Metrics/landing_force_mean ≈ 280` in the
trained run's logs) relative to the other exponential/bounded reward
terms, so a much smaller weight keeps its per-step contribution
comparable in scale rather than dominating the sum.

**`air_time`** (`feet_air_time`, rewards each foot spending
`threshold_min`–`threshold_max` seconds airborne per swing) is wired into
the base reward dict by stock mjlab but — like `body_ang_vel` and
`angular_momentum` — never given a nonzero weight here. See "Inert terms."

## Safety & smoothness

**`dof_pos_limits`** (`mjlab.envs.mdp.joint_pos_limits`, weight -1.0)
penalizes crossing the *soft* joint limits (`soft_joint_pos_limit_factor
=0.9` in `walka_constants.py`'s `WALKA_ARTICULATION` — 90% of the hard
URDF range), so the policy gets pushed back before actually hitting a
hard stop.

**`action_rate_l2`** (`mjlab.envs.mdp.action_rate_l2`, weight -0.1)
penalizes step-to-step change in the *raw* policy output (pre
scale/offset), the standard anti-jitter term — without it, PPO has no
reason to prefer smooth motion over rapidly oscillating position targets
that happen to average out to the right velocity.

**`self_collisions`** (`self_collision_cost`, weight -1.0,
`force_threshold=10N`, added explicitly in `env_cfgs.py` — not in the
stock template) reads the `self_collision` `ContactSensor`
(`subtree`-vs-`subtree` match on `pelvis`, `history_length=4`) and counts
substeps where any self-contact force exceeds 10N. This is the reward-side
half of the kinematic fix in `docs/kinematic_structure_analysis.md` (the
`STRUCTURAL_OVERLAP_PAIRS` excludes handle *always-on* mesh overlap; this
penalizes *genuine* self-collision, e.g. a leg swinging into the other leg
at an extreme angle) — the trained run's `Episode_Reward/self_collisions
≈ -0.0005` (near zero) confirms the fix holds under an actual learned
gait, not just the fixed-base contact sweep it was originally validated
against.

**`angular_momentum`** (`angular_momentum_penalty`, reads the
`root_angmom` `subtreeangmom` sensor from `convert_urdf_to_mjcf.py`) is
meant to discourage large whole-body angular momentum (encouraging, e.g.,
natural arm counter-swing instead of a rigid/spinning gait) — also wired
but inert here; see below.

## Inert terms (weight 0.0)

`body_ang_vel`, `angular_momentum`, and `air_time` are all present in
`RewardManager`'s active-terms table and logged every step, but none of
them has ever been assigned a nonzero weight in
`src/tasks/velocity/config/walka/env_cfgs.py` — they inherit `weight=0.0`
straight from stock mjlab's `make_velocity_env_cfg()` base dict, which
leaves them at 0.0 as an explicit "override per-robot" placeholder (mjlab's
own G1 example sets `body_ang_vel=-0.05`, `angular_momentum=-0.02`,
`air_time=0.0` — i.e. G1 *does* tune the first two and only leaves
`air_time` at zero deliberately, since `foot_gait`'s clock reward already
covers footstep timing). Walka's config never touched any of the three.

This isn't a bug — the trained policy in `docs/images/walka_flat_trained.gif`
already walks with low fall/self-collision rates and near-ceiling tracking
reward without them — but it is unexploited headroom: `body_ang_vel` and
`angular_momentum` in particular are exactly the terms that shape
*upper-body* behavior (arm swing, torso stability) beyond what `pose`'s
static per-joint tolerance captures, so a next tuning pass has an obvious,
already-wired place to start.
