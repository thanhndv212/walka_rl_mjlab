"""Train an RL policy using the mjlab + RSL-RL stack."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()
    cfg.agent.seed = seed
    cfg.env.seed = seed

    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True

    if rank == 0:
        print(f"[INFO] Training with device={device}, seed={seed}, rank={rank}")
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env,
        device=device,
        render_mode="rgb_array" if cfg.video and rank == 0 else None,
    )

    if cfg.agent.resume:
        resume_path = get_checkpoint_path(
            log_dir.parent,
            cfg.agent.load_run,
            cfg.agent.load_checkpoint,
        )
    else:
        resume_path = None

    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)

    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
    args = args or TrainConfig.from_task(task_id)

    log_root = (Path("logs") / "rsl_rl" / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root / log_dir_name

    selected_gpus, num_gpus = select_gpus(args.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task_id, args, log_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir or str(
            log_dir / "torchrunx"
        )

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


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
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
