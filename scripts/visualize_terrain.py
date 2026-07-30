"""Visualize the terrain a task actually trains against.

Loads the terrain generator wired into a registered task's *training* env
config (the same one `train.py` builds the scene from via `Scene._add_terrain`),
so what's rendered here is guaranteed to match what the policy sees during
training - no hand-copied terrain params that can drift out of sync with
env_cfgs.py.

Run with:
  uv run scripts/visualize_terrain.py Walka-Rough
  uv run scripts/visualize_terrain.py Walka-Rough --viewer viser
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import tyro
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.terrains.terrain_entity import TerrainEntity


@dataclass(frozen=True)
class VisualizeConfig:
    viewer: Literal["auto", "native", "viser"] = "auto"


def run_visualize(task_id: str, cfg: VisualizeConfig) -> None:
    env_cfg = load_env_cfg(task_id, play=False)
    terrain_cfg = env_cfg.scene.terrain
    if terrain_cfg is None or terrain_cfg.terrain_generator is None:
        raise SystemExit(
            f"Task '{task_id}' has no terrain generator "
            f"(terrain_type={terrain_cfg.terrain_type if terrain_cfg else None!r}) "
            "- nothing to visualize."
        )

    gen_cfg = terrain_cfg.terrain_generator
    print(
        f"[INFO] {task_id}: {gen_cfg.num_rows}x{gen_cfg.num_cols} grid, "
        f"curriculum={gen_cfg.curriculum}, size={gen_cfg.size}, "
        f"sub_terrains={list(gen_cfg.sub_terrains)}"
    )

    # Mirror Scene._add_terrain so the compiled spec matches training exactly.
    terrain_cfg.num_envs = env_cfg.scene.num_envs
    terrain_cfg.env_spacing = env_cfg.scene.env_spacing
    terrain = TerrainEntity(terrain_cfg, device="cpu")
    model = terrain.spec.compile()

    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"

    if resolved_viewer == "native":
        import mujoco.viewer

        mujoco.viewer.launch(model)
    else:
        _launch_viser(model)


def _launch_viser(model: mujoco.MjModel) -> None:
    import viser
    from mjviser.conversions import merge_geoms

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    terrain_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    terrain_geom_ids = [
        i for i in range(model.ngeom) if model.geom_bodyid[i] == terrain_body_id
    ]
    if not terrain_geom_ids:
        raise SystemExit("No terrain geoms found in the compiled model.")

    mesh = merge_geoms(model, terrain_geom_ids)
    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("/terrain", mesh)
    print(f"[INFO] Viser server running - {len(mesh.faces):,} polygons.")
    while True:
        time.sleep(1.0)


def main() -> None:
    import mjlab.tasks  # noqa: F401

    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(
        VisualizeConfig,
        args=remaining_args,
        default=VisualizeConfig(),
        prog=sys.argv[0] + f" {task_id}",
    )
    run_visualize(task_id, cfg)


if __name__ == "__main__":
    main()
