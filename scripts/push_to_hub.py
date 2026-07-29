"""Publish a promoted Walka velocity checkpoint to the Hugging Face Hub.

Deliberately a separate, manual step — not automatic on every training
checkpoint (unlike the W&B upload in VelocityOnPolicyRunner.save). Only a
checkpoint that's cleared the promotion bar (see
docs/vast_ai_training.md) should get published under a public model repo.

Usage:

    # From a local training run directory:
    uv run python scripts/push_to_hub.py --repo-id <user>/walka-velocity-flat \\
        --run-dir logs/rsl_rl/walka_velocity/2026-07-29_12-00-00

    # From a W&B run (downloads the checkpoint + ONNX export first):
    uv run python scripts/push_to_hub.py --repo-id <user>/walka-velocity-flat \\
        --wandb-run-path <entity>/<project>/<run_id>

`Walka-Flat` and `Walka-Rough` share the same log root
(`logs/rsl_rl/walka_velocity/`, both use `experiment_name="walka_velocity"`
in rl_cfg.py) — --repo-id is what distinguishes which checkpoint a push is
for, e.g. `walka-velocity-flat` vs. `walka-velocity-rough`.

Requires a Hugging Face token: run ``uv run hf auth login`` once first
(paste a token from https://huggingface.co/settings/tokens).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import mjlab
import tyro
import yaml
from huggingface_hub import HfApi, ModelCard, ModelCardData

_CARD_BODY_TEMPLATE = """# Walka velocity — PPO policy

Trained with [walka_rl_mjlab](https://github.com/thanhndv212/walka_rl_mjlab)
(mjlab + RSL-RL PPO) for a Walka biped velocity-tracking task: the policy
commands Walka's 24 joints to track a randomly sampled linear/angular base
velocity command, on either flat ground or randomized rough terrain.

## Files

- `policy.onnx` — the exported policy. Deployment-ready: load with
  `onnxruntime`, no torch/rsl_rl/mjlab needed.
- `model.pt` — the raw RSL-RL checkpoint (actor + critic + optimizer
  state), for resuming training or fine-tuning within walka_rl_mjlab.
- `env.yaml` / `agent.yaml` — the fully-resolved environment and PPO config
  this checkpoint was trained with.

## Provenance

- Iterations: {max_iterations}
- num_envs: {num_envs}
- walka_rl_mjlab commit: {commit_hash}
{wandb_line}

## Usage

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("policy.onnx")
input_name = session.get_inputs()[0].name
obs = np.zeros((1, session.get_inputs()[0].shape[-1]), dtype=np.float32)
(action,) = session.run(None, {{input_name: obs}})
```
"""


@dataclass(frozen=True)
class PushConfig:
    repo_id: str
    """Hugging Face Hub repo id to push to, e.g. 'username/walka-velocity-flat'."""
    run_dir: str | None = None
    """Local training run directory (contains model_*.pt, <run>.onnx, params/)."""
    wandb_run_path: str | None = None
    """W&B run path 'entity/project/run_id' to pull the checkpoint from instead."""
    wandb_checkpoint_name: str | None = None
    """Specific checkpoint filename within the W&B run (default: latest)."""
    private: bool = False
    commit_message: str = "Upload Walka velocity checkpoint"


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Highest-iteration model_<N>.pt in run_dir, mirroring
    mjlab.utils.os.get_checkpoint_path's own selection convention."""
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        return None
    return max(
        checkpoints, key=lambda p: int(re.search(r"model_(\d+)\.pt", p.name).group(1))
    )


def _resolve_run_dir(cfg: PushConfig) -> Path:
    if (cfg.run_dir is None) == (cfg.wandb_run_path is None):
        raise ValueError("Pass exactly one of --run-dir or --wandb-run-path.")

    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    # From W&B: the .pt checkpoint download is covered by an existing mjlab
    # utility; the .onnx export and configs aren't, so fetched here the same way.
    import wandb
    from mjlab.utils.os import get_wandb_checkpoint_path

    log_root = Path("logs") / "rsl_rl" / "walka_velocity"
    checkpoint_path, _ = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
    run_dir = checkpoint_path.parent

    api = wandb.Api()
    wandb_run = api.run(cfg.wandb_run_path)
    for wandb_file in wandb_run.files():
        is_wanted = wandb_file.name.endswith(".onnx") or wandb_file.name.startswith(
            "params/"
        )
        target = run_dir / wandb_file.name
        if is_wanted and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            wandb_file.download(str(run_dir), replace=True)

    return run_dir


def build_model_card(run_dir: Path, cfg: PushConfig) -> str:
    """Render the model card from whatever config/provenance files exist in
    run_dir. Pure function (no network) — see tests/test_push_to_hub.py.
    """
    agent_yaml = run_dir / "params" / "agent.yaml"
    env_yaml = run_dir / "params" / "env.yaml"
    diff_files = (
        sorted((run_dir / "git").glob("*.diff")) if (run_dir / "git").is_dir() else []
    )

    agent_data = yaml.safe_load(agent_yaml.read_text()) if agent_yaml.exists() else {}
    env_data = yaml.safe_load(env_yaml.read_text()) if env_yaml.exists() else {}
    num_envs = (env_data or {}).get("scene", {}).get("num_envs", "unknown")
    max_iterations = (agent_data or {}).get("max_iterations", "unknown")

    commit_hash = "unknown"
    if diff_files:
        match = re.search(r"--- git commit ---\n(\w+)", diff_files[0].read_text())
        if match:
            commit_hash = match.group(1)

    card_data = ModelCardData(
        license="mit",
        library_name="mjlab",
        tags=["reinforcement-learning", "robotics", "mujoco", "biped", "ppo"],
        pipeline_tag="reinforcement-learning",
    )
    wandb_line = f"- W&B run: `{cfg.wandb_run_path}`" if cfg.wandb_run_path else ""
    body = _CARD_BODY_TEMPLATE.format(
        max_iterations=max_iterations,
        num_envs=num_envs,
        commit_hash=commit_hash,
        wandb_line=wandb_line,
    )
    # Not using ModelCard.from_template's Jinja rendering here: it only
    # injects card_data where a literal "{{ card_data }}" placeholder
    # appears, which would collide with the .format() substitution above
    # (str.format treats "{{"/"}}" as escaped literal braces). Front matter
    # is built directly instead.
    return ModelCard(f"---\n{card_data.to_yaml()}\n---\n\n{body}").content


def main(cfg: PushConfig) -> None:
    run_dir = _resolve_run_dir(cfg)

    onnx_path = next(iter(run_dir.glob("*.onnx")), None)
    pt_path = _latest_checkpoint(run_dir)
    if onnx_path is None and pt_path is None:
        raise FileNotFoundError(f"No model_*.pt or *.onnx found in {run_dir}")

    staging = run_dir / "_hub_upload"
    staging.mkdir(exist_ok=True)
    if onnx_path is not None:
        shutil.copy2(onnx_path, staging / "policy.onnx")
    if pt_path is not None:
        shutil.copy2(pt_path, staging / "model.pt")
    for name in ("env.yaml", "agent.yaml"):
        src = run_dir / "params" / name
        if src.exists():
            shutil.copy2(src, staging / name)
    (staging / "README.md").write_text(build_model_card(run_dir, cfg))

    api = HfApi()
    api.create_repo(cfg.repo_id, repo_type="model", private=cfg.private, exist_ok=True)
    api.upload_folder(
        repo_id=cfg.repo_id, folder_path=str(staging), commit_message=cfg.commit_message
    )
    print(f"Pushed to https://huggingface.co/{cfg.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(PushConfig, config=mjlab.TYRO_FLAGS))
