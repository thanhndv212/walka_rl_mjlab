"""Fallen-recovery evaluation harness for Walka-GetUp checkpoints.

Step 0 of docs/get_up_task.md's implementation roadmap. The standing-
probability curriculum's logged ``Episode_Metrics/standing_success`` looked
great at iteration ~700 and only revealed itself as hollow when a checkpoint
was replayed from a genuinely fallen reset, bypassing the curriculum -- the
training-log metric is an average over whatever mix of curriculum-assisted
and genuinely-fallen episodes happened to complete in a given iteration's
rollout window, so it can't be trusted on its own.

This script always resets from the plain fallen-pose distribution: the
``reset_to_standing_curriculum`` event is removed from ``cfg.events``
outright (not just driven to its annealed floor via
``env.common_step_counter``), so the result never depends on the curriculum's
current schedule staying at today's values.

Two metrics are reported:

1. Genuine recovery rate -- both feet in contact and pelvis above
   ``target_height``, held continuously for at least ``hold_seconds``. This
   is the one number that's been trustworthy across every checkpoint tested
   so far (v1.1 through v1.3): it has never produced a false positive, since
   a purely passive fall cannot hold that condition continuously.

2. Peak pelvis height -- **only meaningful relative to the zero-action
   baseline reported alongside it.** Confirmed empirically (2026-08-03): a
   policy applying literally zero action at every step, just settling under
   gravity from the same reset distribution, reaches a mean peak height of
   ~0.53m with ~13% of episodes momentarily topping 0.7m -- indistinguishable
   from every trained checkpoint tested across four different reward designs
   and up to 10,000 training iterations. The peak height number alone
   measures reset-physics noise (the ``ensure_ground_clearance`` correction
   and default-pose joint stiffness settling), not policy skill. Comparing
   the two distributions side by side (and the delta) is the only way this
   metric is informative; never quote the policy's peak-height stats in
   isolation again.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401

TASK_ID = "Walka-GetUp"


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str
    """Path to a model_*.pt checkpoint."""
    num_envs: int = 20
    """Parallel envs per rollout. Each env's reset event samples its own pose
    independently, so num_envs already gives that many independent seeds."""
    num_rollouts: int = 1
    """Repeat the whole num_envs batch this many times (fresh reset each
    time) to accumulate num_envs * num_rollouts total episodes."""
    num_steps: int = 500
    """Policy steps per rollout (env.step_dt seconds each)."""
    target_height: float = 0.7
    """Pelvis height threshold -- matches stand_on_feet's target_height."""
    hold_seconds: float = 1.0
    """Minimum continuous duration the both-feet+height condition must hold
    for an episode to count as a genuine recovery, not a momentary pass-through."""
    sensor_name: str = "feet_ground_contact"
    seed: int = 0
    device: str | None = None
    baseline: bool = True
    """Also roll out a matched zero-action baseline (same seeds -> same
    initial fallen poses) so peak-height numbers are reported in context
    instead of in a vacuum. Costs a second rollout pass; only disable this
    for a quick smoke check where the peak-height numbers won't be read."""


class Rollout:
    """One (policy-or-zero-action) sweep of ``cfg.num_rollouts x cfg.num_envs``
    episodes against a shared, already-constructed env. Kept as a class only
    to hold the env/robot/sensor handles -- ``run`` has no other state and
    can be called multiple times (e.g. once per policy) against the same env.
    """

    def __init__(self, wrapped, robot, sensor, cfg: EvalConfig) -> None:
        self._wrapped = wrapped
        self._robot = robot
        self._sensor = sensor
        self._cfg = cfg
        self._step_dt = wrapped.unwrapped.step_dt
        self._num_actions = wrapped.unwrapped.action_manager.total_action_dim

    def _height_and_condition(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self._cfg
        h = self._robot.data.root_link_pos_w[:, 2]
        both_feet = (self._sensor.data.current_contact_time > 0).all(dim=1)
        return h, both_feet & (h > cfg.target_height)

    def run(self, policy=None) -> tuple[list[float], list[bool]]:
        """policy=None drives every step with zero action (the baseline)."""
        cfg = self._cfg
        step_dt = self._step_dt
        zero_action = torch.zeros(cfg.num_envs, self._num_actions)

        peak_heights: list[float] = []
        successes: list[bool] = []

        for rollout_idx in range(cfg.num_rollouts):
            torch.manual_seed(cfg.seed + rollout_idx)
            obs_dict, _ = self._wrapped.reset()

            h, condition = self._height_and_condition()
            peak_h = h.clone()
            streak = torch.where(condition, step_dt, 0.0) * torch.ones_like(h)
            success = streak >= cfg.hold_seconds

            for _ in range(cfg.num_steps):
                if policy is not None:
                    with torch.no_grad():
                        action = policy(obs_dict)
                else:
                    action = zero_action
                obs_dict, _, dones, _ = self._wrapped.step(action)
                done_mask = dones.bool()

                # Close out episodes that ended this step using the
                # peak/streak accumulated *before* this step -- by the time
                # dones[i] is True, env.step() has already auto-reset env i,
                # so h/condition below reflect the *new* episode's state,
                # not the one that just ended.
                if torch.any(done_mask):
                    for i in done_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                        peak_heights.append(peak_h[i].item())
                        successes.append(bool(success[i].item()))

                h, condition = self._height_and_condition()
                peak_h = torch.where(done_mask, h, torch.maximum(peak_h, h))
                streak = torch.where(
                    condition,
                    torch.where(
                        done_mask, step_dt * torch.ones_like(h), streak + step_dt
                    ),
                    torch.zeros_like(h),
                )
                new_success = streak >= cfg.hold_seconds
                success = torch.where(done_mask, new_success, success | new_success)

            # Close out whatever's still in-flight at the end of the rollout.
            for i in range(cfg.num_envs):
                peak_heights.append(peak_h[i].item())
                successes.append(bool(success[i].item()))

        return peak_heights, successes


def _print_stats(label: str, peak_heights: list[float], successes: list[bool]) -> None:
    n = len(successes)
    n_success = sum(successes)
    print(
        f"[{label}] genuine recovery: {n_success / n:.1%} ({n_success}/{n})  |  "
        f"peak height mean={statistics.mean(peak_heights):.3f} "
        f"median={statistics.median(peak_heights):.3f} "
        f"min={min(peak_heights):.3f} max={max(peak_heights):.3f}"
    )
    fracs = []
    for thresh in (0.3, 0.5, 0.7, 0.8):
        frac = sum(p > thresh for p in peak_heights) / n
        fracs.append(f">{thresh}m: {frac:.1%}")
    print(f"[{label}] peak-height fractions -- " + "  ".join(fracs))


def run_eval(cfg: EvalConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    removed = env_cfg.events.pop("reset_to_standing_curriculum", None)
    if removed is None:
        print(
            "[WARN] 'reset_to_standing_curriculum' not found in cfg.events -- "
            "either already plain-fallen-only, or the event was renamed. "
            "Double check this run isn't silently curriculum-assisted."
        )
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    robot = env.scene["robot"]
    sensor = env.scene.sensors[cfg.sensor_name]
    rollout = Rollout(wrapped, robot, sensor, cfg)

    print(f"Checkpoint: {cfg.checkpoint}")
    print(
        f"Episodes per pass: {cfg.num_rollouts * cfg.num_envs} "
        f"({cfg.num_rollouts} rollout(s) x {cfg.num_envs} envs), "
        f"condition: both feet + h>{cfg.target_height}m held >={cfg.hold_seconds}s"
    )

    policy_peaks, policy_successes = rollout.run(policy=policy)
    _print_stats("policy", policy_peaks, policy_successes)

    if cfg.baseline:
        # Same seed sequence -> same reset-sampled initial poses as the
        # policy pass above, so this is a paired comparison: only the
        # actions differ (zero vs. learned).
        baseline_peaks, baseline_successes = rollout.run(policy=None)
        _print_stats("zero-action baseline", baseline_peaks, baseline_successes)
        delta = statistics.mean(policy_peaks) - statistics.mean(baseline_peaks)
        print(
            f"[delta] policy mean peak height minus baseline mean peak height: "
            f"{delta:+.3f}m -- this, not the policy's raw peak height, is the "
            "actual skill signal."
        )

    env.close()


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    run_eval(cfg)


if __name__ == "__main__":
    main()
