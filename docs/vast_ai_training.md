# Training Walka velocity tasks on vast.ai

A real training run — thousands of parallel envs, full `max_iterations` — on
a rented GPU, tracked via W&B, then played back locally in sim, and
published to the Hugging Face Hub once it clears a promotion bar. Mirrors
the workflow in `soarm_mjlab/docs/vast_ai_training.md`; adapted here for
`walka_rl_mjlab`'s two velocity tasks (`Walka-Flat`, `Walka-Rough`) instead
of an arm-reaching task.

`walka_rl_mjlab` is self-contained for this purpose: the generated MJCF +
meshes live in `src/assets/robots/walka/xmls/`, committed in-repo, so the
rented box only ever needs `walka_rl_mjlab` itself.

## Prerequisites

- A [vast.ai](https://vast.ai) account with billing set up (credits or a
  card on file) — **check your balance before renting**
  (`vastai show user`); a negative or zero balance will fail to launch.
- A W&B account. If this machine is already logged in (`~/.netrc` has an
  `api.wandb.ai` entry), the rented box can reuse the same key
  non-interactively (step 3); otherwise the rented box needs its own
  `wandb login` — get an API key from <https://wandb.ai/authorize>.
- A Hugging Face account, only needed on whichever machine runs
  `scripts/push_to_hub.py` (this Mac, in the flow below — not the rented
  box). Authenticate once with `uv run hf auth login` (paste a token from
  <https://huggingface.co/settings/tokens>).
- Nothing else — `walka_rl_mjlab` is public
  (<https://github.com/thanhndv212/walka_rl_mjlab>), no deploy key or token
  needed to clone it.

## 1. Rent an instance

On the [vast.ai console](https://cloud.vast.ai/create/), search offers and
pick:

- **GPU**: unlike an arm-reaching task, this is a full biped locomotion task
  with a rough-terrain generator, a raycast height-scan sensor
  (`terrain_scan`, 17×11 = 187-dim), and a per-foot ring-pattern height
  sensor (`foot_height_scan`) — real per-step GPU compute beyond the policy
  network itself (still a small MLP, 512×256×128). Treat this closer to
  mjlab's own G1 humanoid example than to a tabletop arm task: an RTX
  4090/5090 or A5000/A6000 is a reasonable starting tier. **Verified**: a
  full `Walka-Flat` run (10001 iterations, `num_envs=4096`) completed on a
  rented RTX 4090 (24GB) in ~1h54m, no OOM, no preemption issues — watch the
  printed `Steps per second` for the first ~30s of the real run (step 5) and
  adjust `--env.scene.num-envs` from there if your offer's GPU differs.
  `Walka-Rough` (terrain generator + raycasting) hasn't been run on GPU yet —
  expect lower throughput per env than `Walka-Flat` at the same `num_envs`.
- **VRAM**: start around 16–24GB. `Walka-Rough`'s raycast sensor and
  terrain-generator geometry cost more VRAM per env than `Walka-Flat`
  (no terrain mesh, no raycasting) — if you're unsure, do the first real run
  on `Walka-Flat` to get a per-env memory/compute baseline before committing
  to `Walka-Rough` at the same `num_envs`.
- **vCPUs**: 4–8. Physics and PPO both run on the GPU via
  `mujoco_warp`/torch; the CPU is orchestration only.
- **RAM**: 16–32GB is comfortable.
- **Image/template**: any template with CUDA 12.8+ and a recent Ubuntu
  (vast.ai's own "PyTorch" template, or `nvidia/cuda:12.8.1-devel-ubuntu22.04`)
  works — `uv sync --extra cu128` installs torch itself. Bare CUDA *devel*
  images don't include `libEGL`/`libGL`, which `import mujoco` needs even
  when nothing is rendering (an unconditional import inside the package) —
  `setup_remote.sh` now installs `libegl1 libgl1 libglx0 libopengl0`
  automatically when they're missing, so this is handled either way.
- **Disk**: 30GB is comfortable (repo + meshes are a few MB; the rest is
  torch/CUDA wheels + checkpoints).
- **Interruptible vs. on-demand**: interruptible is cheaper but can be
  preempted; checkpoints save every `save_interval` iterations (100, see
  `rl_cfg.py`) and training resumes from the last one with
  `--agent.resume`. **Preemption can be indefinite** — if a restart queues
  for a long time, rent a fresh instance instead (the repo re-clones and
  `uv sync`s in minutes) and `vastai destroy instance <id> -y` the stuck one.
- **Check `disk_bw`** when comparing offers, not just price/GPU — a host
  with low disk bandwidth can make `uv sync --extra cu128`'s multi-GB
  torch/CUDA wheel unpack take 30+ minutes even on a fast network link.

Launch it and wait for the instance to show **running** in the Instances tab.

## 2. Connect

Copy the SSH command from the instance's card in the vast.ai console (or use
their web terminal):

```bash
ssh -p <PORT> root@<HOST>
```

## 3. One-time setup (+ automated W&B auth)

```bash
# On the Mac (already has ~/.netrc from being logged in locally):
WANDB_API_KEY=$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')
ssh -p <PORT> root@<HOST> "WANDB_API_KEY=$WANDB_API_KEY bash -s" < scripts/setup_remote.sh
```

This installs `uv`, clones `walka_rl_mjlab`, syncs the `cu128` extra, and
runs `wandb login` non-interactively when `WANDB_API_KEY` is set.

Without the env var:

```bash
curl -LsSf https://raw.githubusercontent.com/thanhndv212/walka_rl_mjlab/main/scripts/setup_remote.sh | bash
```

Or by hand:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/thanhndv212/walka_rl_mjlab.git
cd walka_rl_mjlab
uv sync --locked --extra cu128 --group dev
```

## 4. Authenticate W&B

Skip this if you used the `WANDB_API_KEY` one-liner in step 3.

```bash
uv run wandb login
# paste the API key from https://wandb.ai/authorize
```

## 5. Launch training (in tmux — don't skip this)

```bash
tmux new -s train
cd walka_rl_mjlab
uv run --extra cu128 --group dev python scripts/train.py Walka-Flat --env.scene.num-envs=4096
```

Always repeat `--extra cu128 --group dev` on every `uv run` invocation that
touches `torch` (train/play/push_to_hub), not just the one-time setup sync —
see Troubleshooting below for what happens if you drop it.

Start with `Walka-Flat` to get a clean steps/sec and stability baseline
before moving to `Walka-Rough` (heavier per-env cost from the terrain
generator + raycasting, see step 1). Swap the task ID for `Walka-Rough`
once you have that baseline. `max_iterations` (10001, from `rl_cfg.py`) and
`num_envs` are the main levers — scale `num_envs` down if the GPU reports
OOM (watch the first few seconds of output).

Detach with **Ctrl-b d**; reattach with `tmux attach -t train`. Check
progress non-invasively with `tmux capture-pane -t train -p | tail -30`.

## 6. Monitor

`train.py`'s first lines include a W&B run URL — open it for the reward
curve and all `Episode_Reward/*` / `Episode_Termination/*` / `Metrics/*`
scalars live. Tensorboard logs also land in
`logs/rsl_rl/walka_velocity/` if you'd rather tunnel:

```bash
# On the Mac:
ssh -p <PORT> -L 6006:localhost:6006 root@<HOST>
# On the rented box, in a second tmux window:
uv run tensorboard --logdir logs/rsl_rl/walka_velocity --host 0.0.0.0
```

## 7. Decide the promotion bar *before* looking at the finished curve

Pick the bar now, not after seeing the number. Reasonable starting points
using metrics already logged by this task
(`src/tasks/velocity/config/walka/env_cfgs.py`):

- `Episode_Termination/fell_over` and `Episode_Termination/out_of_terrain_bounds`
  both small (e.g. combined < 0.05) over the last ~100 logged episodes —
  the robot is staying upright and on the terrain, not just accumulating
  reward some other way.
- `Episode_Reward/track_linear_velocity` and `Episode_Reward/track_angular_velocity`
  each trending toward their per-step weight ceiling (2.0), not plateauing
  well below it.
- `Episode_Reward/self_collisions` (weight -1.0) staying near 0 — confirms
  the kinematic fix in `docs/kinematic_structure_analysis.md` is holding up
  under real gait, not just the fixed-base/contact-sweep checks it was
  validated against.

If a run doesn't clear this, that's a finding (reward shaping, more
iterations, curriculum tuning) — not a reason to lower the bar after the fact.

## 8. Retrieve the checkpoint

Both the `.pt` checkpoint and the `.onnx` export are uploaded to the W&B run
automatically (`VelocityOnPolicyRunner.save`, mirrors `soarm_mjlab`'s
`ReachOnPolicyRunner`) — no manual copy needed.

```bash
uv run python scripts/play.py Walka-Flat \
    --wandb-run-path <entity>/<project>/<run_id>
```

Or pull the raw files directly:

```bash
scp -P <PORT> root@<HOST>:walka_rl_mjlab/logs/rsl_rl/walka_velocity/*/model_*.pt .
```

## 9. Play it back locally

Same command as step 8 — `scripts/play.py` picks a native MuJoCo window
when a display is available. Use `--video` for an mp4, or
`--no-terminations` for full rollouts without early episode cutoffs.

## 10. Publish the checkpoint to the Hugging Face Hub

Only once it clears the promotion bar from step 7:

```bash
uv run python scripts/push_to_hub.py \
    --repo-id <your-hf-username>/walka-velocity-flat \
    --wandb-run-path <entity>/<project>/<run_id>
```

(Use `walka-velocity-rough` for a `Walka-Rough` checkpoint.) Downloads the
ONNX export + configs from the W&B run if not already cached locally,
generates a model card with training provenance, and pushes
`policy.onnx` + `model.pt` + `env.yaml`/`agent.yaml` + `README.md` to a new
or existing HF model repo. Add `--private` to keep it unlisted. If you
already have the run directory locally, `--run-dir <path>` skips the W&B
download.

## 11. Shut the instance down

vast.ai bills while an instance is **running**, and *stopped* instances
still bill for disk. Once you have the checkpoint and are done iterating,
**destroy** the instance (`vastai destroy instance <id>` or via the
console) rather than just stopping it.

## Troubleshooting

- **`ImportError: libcudnn.so.9: cannot open shared object file`** (or any
  other CUDA `.so` missing) right at `import torch`: `cu128` isn't a default
  extra in `pyproject.toml`, so a bare `uv run python ...` (no
  `--extra cu128 --group dev`) re-resolves against the *unconstrained*
  dependency set and silently swaps in a different, unpinned `torch` build
  (observed: `torch==2.13.0` with `nvidia-*-cu13` sibling packages instead
  of the locked `torch==2.11.0+cu128`). Worse, bouncing between the two
  resolutions a few times (e.g. `uv sync --extra cu128` → bare `uv run` →
  `uv sync --extra cu128` again) can leave the venv in a broken half-state —
  `nvidia_cudnn_cu12`'s `dist-info` present but its actual `lib/libcudnn.so*`
  files missing, so `uv` considers it "installed" and won't refetch it. Fix:
  `rm -rf .venv && uv sync --locked --extra cu128 --group dev` for a clean
  rebuild, then always pass `--extra cu128 --group dev` on every subsequent
  `uv run` that touches `torch`, with no exceptions.
- **CUDA/driver mismatch at `uv sync`**: pick a different offer — vast.ai
  lists each host's driver version; it needs to support CUDA 12.8.
- **OOM during `env` construction or the first PPO update**: lower
  `--env.scene.num-envs`. `Walka-Rough` needs a lower ceiling than
  `Walka-Flat` at the same GPU (terrain + raycasting overhead).
- **Training stops when the SSH connection drops**: you skipped the tmux
  step — always launch inside `tmux`/`screen`.
- **`wandb: ERROR ... 401`**: the API key wasn't picked up — rerun
  `uv run wandb login` inside the repo's venv (`uv run`, not a bare
  `wandb` on the system Python).
- **`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`
  (or any `libEGL`/`libGL` import error) right at startup**, before
  training even gets to argument parsing — a bare CUDA *devel* image is
  missing `libegl1`/`libgl1`, which `import mujoco` needs unconditionally
  (regardless of `--video`). `setup_remote.sh` installs these
  automatically now; if you ran it from an older checkout, `apt-get
  install -y libegl1 libgl1 libglx0 libopengl0` and retry. **This crashes
  before `wandb.init()` runs** — if a training run "did nothing" and never
  showed up in W&B, check this first (`tail` the tmux pane/log), not W&B
  auth.
- **`push_to_hub.py` fails with a 401/permission error**: not logged in (or
  the token lacks write access) on the machine running it — rerun
  `uv run hf auth login` there.
- **`vastai` CLI reports insufficient balance / instance won't launch**:
  check `vastai show user` — a negative or zero balance blocks new rentals
  even if an offer shows as available; top up before renting.
- **Forgot to destroy the instance after a run**: if you're driving the
  instance from a controlling machine (not just an interactive SSH
  session), a small polling loop that stops the instance once GPU
  utilization and tmux sessions have been idle for N minutes
  (`nvidia-smi --query-gpu=utilization.gpu`, `tmux list-sessions`, then
  `vastai stop instance <id>`) is a cheap safety net against paying for
  idle GPU time after training finishes. Still bills for disk while
  stopped — `vastai destroy instance <id>` once you're really done.
