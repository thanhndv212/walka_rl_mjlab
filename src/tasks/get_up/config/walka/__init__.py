from mjlab.tasks.registry import register_mjlab_task

from src.tasks.get_up.rl import GetUpOnPolicyRunner

from .env_cfgs import walka_get_up_env_cfg
from .rl_cfg import walka_get_up_ppo_runner_cfg

register_mjlab_task(
    task_id="Walka-GetUp",
    env_cfg=walka_get_up_env_cfg(),
    play_env_cfg=walka_get_up_env_cfg(play=True),
    rl_cfg=walka_get_up_ppo_runner_cfg(),
    runner_cls=GetUpOnPolicyRunner,
)