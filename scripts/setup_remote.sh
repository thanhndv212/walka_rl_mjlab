#!/usr/bin/env bash
# One-time setup for a rented GPU box (vast.ai or similar): installs uv,
# clones walka_rl_mjlab (self-contained — the generated MJCF + meshes are
# committed in-repo under src/assets/robots/walka/xmls/), syncs the cu128
# extra, and (if WANDB_API_KEY is set) authenticates W&B non-interactively.
#
# See docs/vast_ai_training.md for the full step-by-step guide this
# script is one step of.
#
# Usage (on the remote box):
#   curl -LsSf https://raw.githubusercontent.com/thanhndv212/walka_rl_mjlab/main/scripts/setup_remote.sh | bash
#
# To also skip the manual `wandb login` step (recommended when renting
# instances often — the key is short-lived per rental, not committed
# anywhere): export WANDB_API_KEY before piping into bash, e.g. from the
# local machine:
#   WANDB_API_KEY=$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')
#   ssh -p <PORT> root@<HOST> "WANDB_API_KEY=$WANDB_API_KEY bash -s" \
#     < scripts/setup_remote.sh

set -euo pipefail

REPO_URL="https://github.com/thanhndv212/walka_rl_mjlab.git"
REPO_DIR="walka_rl_mjlab"

# Bare CUDA devel images (e.g. nvidia/cuda:12.8.1-devel-ubuntu22.04) don't
# include libEGL/libGL — `import mujoco` fails at import time (unconditional
# renderer import) without them, well before any actual rendering is
# attempted. Templates that already bundle a desktop/OpenGL stack (vast.ai's
# "PyTorch" template, etc.) already have these; the install is a harmless
# no-op there.
if command -v apt-get &>/dev/null && ! ldconfig -p 2>/dev/null | grep -q libEGL.so; then
  echo "==> Installing libEGL/libGL (missing on bare CUDA images, needed just to import mujoco)"
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1 libglx0 libopengl0
fi

if ! command -v uv &>/dev/null; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "==> $REPO_DIR already cloned, pulling latest"
  git -C "$REPO_DIR" pull
else
  echo "==> Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "==> Syncing cu128 extra (--locked: fails if uv.lock is stale)"
uv sync --locked --extra cu128 --group dev

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "==> Authenticating W&B from WANDB_API_KEY"
  uv run wandb login "$WANDB_API_KEY"
  wandb_step="1. W&B already authenticated (WANDB_API_KEY was set)."
else
  wandb_step="1. Authenticate W&B (paste the API key from https://wandb.ai/authorize):
       cd walka_rl_mjlab && uv run wandb login"
fi

echo ""
echo "==> Setup complete. Next steps:"
echo ""
echo "  $wandb_step"
cat <<'EOF'

  2. Start a tmux session so training survives an SSH disconnect:
       tmux new -s train

  3. Launch training (inside tmux) — start with Walka-Flat to get a
     steps/sec and stability baseline before the heavier Walka-Rough:
       uv run python scripts/train.py Walka-Flat --env.scene.num-envs=4096

  4. Detach with Ctrl-b d; reattach later with: tmux attach -t train

See docs/vast_ai_training.md for monitoring, retrieving the checkpoint, and
shutting the instance down when done.
EOF
