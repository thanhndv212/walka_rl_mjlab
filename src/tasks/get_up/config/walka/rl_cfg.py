"""RSL-RL config for Walka get-up tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def walka_get_up_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="walka_get_up",
        # 1500 (not 100): every save_interval uploads a full checkpoint to
        # W&B, which has a hard artifact-storage quota -- a handful of
        # full-length runs at the old 100 (~150 checkpoints/run) exhausted
        # it. W&B is short-term tracking; scripts/push_to_hub.py is the
        # long-term store for a checkpoint actually worth keeping (see
        # docs/vast_ai_training.md).
        save_interval=1500,
        num_steps_per_env=24,
        max_iterations=15001,
    )