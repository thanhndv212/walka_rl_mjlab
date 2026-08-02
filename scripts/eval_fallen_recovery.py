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
current schedule staying at today's values. It reports the fraction of
rollouts that reach a genuine, *sustained* standing condition -- both feet in
contact and pelvis above ``target_height``, held continuously for at least
``hold_seconds`` -- plus the peak pelvis height distribution, which is the
one number that should be trusted from here on instead of the training log.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

import torch
import tyro

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

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
    step_dt = env.step_dt

    def height_and_condition() -> tuple[torch.Tensor, torch.Tensor]:
        h = robot.data.root_link_pos_w[:, 2]
        both_feet = (sensor.data.current_contact_time > 0).all(dim=1)
        return h, both_feet & (h > cfg.target_height)

    peak_heights: list[float] = []
    successes: list[bool] = []

    for rollout in range(cfg.num_rollouts):
        torch.manual_seed(cfg.seed + rollout)
        obs_dict, _ = wrapped.reset()

        h, condition = height_and_condition()
        peak_h = h.clone()
        streak = torch.where(condition, step_dt, 0.0) * torch.ones_like(h)
        success = streak >= cfg.hold_seconds

        for _ in range(cfg.num_steps):
            with torch.no_grad():
                action = policy(obs_dict)
            obs_dict, _, dones, _ = wrapped.step(action)
            done_mask = dones.bool()

            # Close out episodes that ended this step using the peak/streak
            # accumulated *before* this step -- by the time dones[i] is True,
            # env.step() has already auto-reset env i, so h/condition below
            # reflect the *new* episode's state, not the one that just ended.
            if torch.any(done_mask):
                for i in done_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    peak_heights.append(peak_h[i].item())
                    successes.append(bool(success[i].item()))

            h, condition = height_and_condition()
            peak_h = torch.where(done_mask, h, torch.maximum(peak_h, h))
            streak = torch.where(
                condition,
                torch.where(done_mask, step_dt * torch.ones_like(h), streak + step_dt),
                torch.zeros_like(h),
            )
            new_success = streak >= cfg.hold_seconds
            success = torch.where(done_mask, new_success, success | new_success)

        # Close out whatever's still in-flight at the end of the rollout.
        for i in range(cfg.num_envs):
            peak_heights.append(peak_h[i].item())
            successes.append(bool(success[i].item()))

    n = len(successes)
    n_success = sum(successes)
    print(f"Checkpoint: {cfg.checkpoint}")
    print(f"Episodes evaluated: {n} ({cfg.num_rollouts} rollout(s) x {cfg.num_envs} envs)")
    print(
        f"Genuine recovery rate (both feet + h>{cfg.target_height}m held "
        f">={cfg.hold_seconds}s, from plain fallen resets): "
        f"{n_success / n:.1%} ({n_success}/{n})"
    )
    print(
        "Peak pelvis height per episode -- "
        f"mean={statistics.mean(peak_heights):.3f}  "
        f"median={statistics.median(peak_heights):.3f}  "
        f"min={min(peak_heights):.3f}  max={max(peak_heights):.3f}"
    )
    for thresh in (0.3, 0.5, 0.7, 0.8):
        frac = sum(p > thresh for p in peak_heights) / n
        print(f"  fraction reaching peak height > {thresh}m: {frac:.1%}")

    env.close()


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    run_eval(cfg)


if __name__ == "__main__":
    main()
