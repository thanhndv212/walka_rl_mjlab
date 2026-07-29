# Walka RL Mjlab

![Trained Walka-Flat policy: a stable forward walking gait in sim](docs/images/walka_flat_trained.gif)

*Trained Walka-Flat policy: a stable forward walking gait in sim*

`Walka-Flat` has been trained end-to-end on a rented vast.ai RTX 4090
(10001 iterations, `num_envs=4096`, ~1h54m, no OOM) — that's the gait
above. `Walka-Rough` is registered but not yet trained.

## Install

```bash
make sync-cpu   # dev machine without a GPU: uv sync --extra cpu --group dev
make sync       # GPU box, CUDA 12.8: uv sync --extra cu128 --group dev
```

## Usage

```bash
# List all registered tasks.
uv run python scripts/list_envs.py

# Train a policy (swap Walka-Flat for Walka-Rough to train on terrain).
uv run python scripts/train.py Walka-Flat --env.scene.num-envs=4096

# Play back a trained checkpoint in the viewer.
uv run python scripts/play.py Walka-Flat --checkpoint-file logs/rsl_rl/walka_velocity/DATE/model_N.pt

# Publish a promoted checkpoint to the Hugging Face Hub.
uv run python scripts/push_to_hub.py --repo-id <user>/walka-velocity-flat --wandb-run-path <entity>/<project>/<run_id>
```

No local GPU needed — see `docs/vast_ai_training.md` for the full
rented-GPU workflow (instance selection, monitoring, promotion bar,
Hugging Face Hub publish).

## Docs

- `docs/reward_design.md` — what each reward term does and how it shapes the gait.
- `docs/vast_ai_training.md` — rented-GPU training workflow.

Repo: <https://github.com/thanhndv212/walka_rl_mjlab>

## Acknowledgements

- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) —
  this repo's task/script structure mirrors it, and the gait-clock `phase`
  observation and `feet_gait`/`stand_still` rewards
  (`src/tasks/velocity/mdp/`) are ported from its local velocity-task fork.
