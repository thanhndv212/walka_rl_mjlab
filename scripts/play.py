"""Play a trained policy in a selected mjlab task."""

import os
import sys
from dataclasses import asdict, dataclass
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


def run_play(task_id: str, cfg: PlayConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}

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

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
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
