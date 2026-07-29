from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    walka_flat_env_cfg,
    walka_rough_env_cfg,
)
from .rl_cfg import walka_ppo_runner_cfg

register_mjlab_task(
    task_id="Walka-Rough",
    env_cfg=walka_rough_env_cfg(),
    play_env_cfg=walka_rough_env_cfg(play=True),
    rl_cfg=walka_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Walka-Flat",
    env_cfg=walka_flat_env_cfg(),
    play_env_cfg=walka_flat_env_cfg(play=True),
    rl_cfg=walka_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
