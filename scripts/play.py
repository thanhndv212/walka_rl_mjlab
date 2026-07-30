"""Play a trained policy in a selected mjlab task."""

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_E, KEY_J, KEY_L, KEY_Q, KEY_S, KEY_W, KEY_X


def _override_twist_command(env, lin_vel_x: float, lin_vel_y: float, ang_vel_z: float):
    """Monkey-patch the 'twist' command term's compute() to hold a fixed value.

    Runs on the main sim thread each env.step() (compute() is called from
    inside the command manager, not the viewer/render thread), overriding
    whatever the task's UniformVelocityCommand would have resampled. Applies
    to env index 0 only (play's default num_envs=1). Returns the mutable
    state dict so callers (e.g. keyboard steering) can update it live.
    """
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    steer_state = {
        "lin_vel_x": lin_vel_x,
        "lin_vel_y": lin_vel_y,
        "ang_vel_z": ang_vel_z,
    }

    original_compute = cmd_term.compute

    def steering_compute(dt: float) -> None:
        original_compute(dt)
        cmd_term.vel_command_b[0, 0] = steer_state["lin_vel_x"]
        cmd_term.vel_command_b[0, 1] = steer_state["lin_vel_y"]
        cmd_term.vel_command_b[0, 2] = steer_state["ang_vel_z"]

    cmd_term.compute = steering_compute
    return steer_state


def _make_steering_key_callback(env, step: float):
    """WASD/QE keyboard override for the 'twist' velocity command.

    W/S: lin_vel_x (forward/back). J/L: lin_vel_y (strafe left/right).
    Q/E: ang_vel_z (turn left/right). X: zero all three. Each keypress nudges
    the command by `step` and clamps to the task's trained command ranges, so
    the policy never sees an out-of-distribution command.
    """
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    ranges = cmd_term.cfg.ranges
    steer_state = _override_twist_command(env, 0.0, 0.0, 0.0)

    def _clamp(name: str, value: float, bounds: tuple[float, float]) -> None:
        steer_state[name] = max(bounds[0], min(bounds[1], value))

    def key_callback(key: int) -> None:
        if key == KEY_W:
            _clamp("lin_vel_x", steer_state["lin_vel_x"] + step, ranges.lin_vel_x)
        elif key == KEY_S:
            _clamp("lin_vel_x", steer_state["lin_vel_x"] - step, ranges.lin_vel_x)
        elif key == KEY_J:
            _clamp("lin_vel_y", steer_state["lin_vel_y"] + step, ranges.lin_vel_y)
        elif key == KEY_L:
            _clamp("lin_vel_y", steer_state["lin_vel_y"] - step, ranges.lin_vel_y)
        elif key == KEY_Q:
            _clamp("ang_vel_z", steer_state["ang_vel_z"] + step, ranges.ang_vel_z)
        elif key == KEY_E:
            _clamp("ang_vel_z", steer_state["ang_vel_z"] - step, ranges.ang_vel_z)
        elif key == KEY_X:
            steer_state["lin_vel_x"] = 0.0
            steer_state["lin_vel_y"] = 0.0
            steer_state["ang_vel_z"] = 0.0
        else:
            return
        print(
            f"[steer] lin_vel_x={steer_state['lin_vel_x']:+.2f} "
            f"lin_vel_y={steer_state['lin_vel_y']:+.2f} "
            f"ang_vel_z={steer_state['ang_vel_z']:+.2f}"
        )

    print(
        "[INFO] Keyboard steering enabled: W/S=fwd/back  J/L=strafe  "
        "Q/E=turn  X=stop"
    )
    return key_callback


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    keyboard_steer: bool = False
    """Drive the 'twist' velocity command with WASD/QE instead of letting it
    randomize. Native viewer only."""
    steer_step: float = 0.1
    """Velocity increment per keypress when --keyboard-steer is set."""
    terrain: str | None = None
    """Restrict the terrain generator to a single named sub-terrain (e.g.
    'random_rough'), so every patch in the grid is that type. Only applies
    to tasks with a terrain generator."""
    terrain_difficulty: float = 0.3
    """Fixed difficulty (0-1) for --terrain patches, instead of the default
    random per-patch sampling over the full (0.0, 1.0) range — curriculum=False
    mode samples independently per patch, which can hand a demo rollout an
    unreasonably hard (e.g. near-vertical) patch it was never meant to solve
    at a fixed forward speed."""
    forward_speed: float | None = None
    """Force a fixed lin_vel_x twist command (lin_vel_y/ang_vel_z=0) instead
    of letting it randomize. Useful for demo recordings on terrains with a
    flat spawn platform (e.g. stairs/slope), so the robot deliberately walks
    off it instead of idling. Ignored if --keyboard-steer is also set."""


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}

    if cfg.terrain is not None:
        tgen = env_cfg.scene.terrain.terrain_generator if env_cfg.scene.terrain else None
        if tgen is None:
            raise SystemExit(f"Task '{task_id}' has no terrain generator to restrict.")
        if cfg.terrain not in tgen.sub_terrains:
            raise SystemExit(
                f"Unknown terrain '{cfg.terrain}'. Available: {list(tgen.sub_terrains)}"
            )
        chosen = replace(tgen.sub_terrains[cfg.terrain], proportion=1.0)
        tgen.sub_terrains = {cfg.terrain: chosen}
        tgen.num_cols = 1
        tgen.curriculum = False
        tgen.difficulty_range = (cfg.terrain_difficulty, cfg.terrain_difficulty)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    dummy_mode = cfg.agent in {"zero", "random"}
    trained_mode = not dummy_mode

    log_dir: Path | None = None
    resume_path: Path | None = None
    if trained_mode:
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        else:
            log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
            resume_path = get_checkpoint_path(
                log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        log_dir = resume_path.parent
        print(f"[INFO] Loading checkpoint: {resume_path}")

    render_mode = "rgb_array" if (trained_mode and cfg.video) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if trained_mode and cfg.video:
        assert log_dir is not None
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if dummy_mode:
        action_shape = env.unwrapped.action_space.shape
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        assert resume_path is not None
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device=device)

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    key_callback = None
    if cfg.keyboard_steer:
        if resolved_viewer != "native":
            print(
                "[WARN] --keyboard-steer only wires into the native viewer; "
                f"ignoring it for viewer={resolved_viewer!r}."
            )
        elif "twist" not in env.unwrapped.command_manager.active_terms:
            print("[WARN] No 'twist' command term on this task; ignoring --keyboard-steer.")
        else:
            key_callback = _make_steering_key_callback(env, cfg.steer_step)
    elif cfg.forward_speed is not None:
        if "twist" not in env.unwrapped.command_manager.active_terms:
            print("[WARN] No 'twist' command term on this task; ignoring --forward-speed.")
        else:
            _override_twist_command(env, cfg.forward_speed, 0.0, 0.0)

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy, key_callback=key_callback).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


def main() -> None:
    import mjlab.tasks

    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
