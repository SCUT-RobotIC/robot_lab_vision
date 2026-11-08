# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

try:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg
except ImportError:
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg


def stony_road_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the stone-road height field used by the RC rough-stones task."""
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = max(int(cfg.noise_step / cfg.vertical_scale), 1)
    height_range = np.arange(height_min, height_max + height_step, height_step)

    if cfg.downsampled_scale == cfg.horizontal_scale:
        # Sample the final grid directly. The mesh needs one more vertex than the
        # number of horizontal intervals; repeating a 160x160 grid to 161x161
        # would otherwise turn every sample into an unintended 2x2 plateau.
        heights = np.random.choice(height_range, size=(width_pixels, length_pixels))
    else:
        heights = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
        repeat_x = max(int(np.ceil(width_pixels / width_downsampled)), 1)
        repeat_y = max(int(np.ceil(length_pixels / length_downsampled)), 1)
        heights = np.repeat(np.repeat(heights, repeat_x, axis=0), repeat_y, axis=1)
        heights = heights[:width_pixels, :length_pixels]
    heights = heights.astype(np.int16)

    wall_pixels = max(int(np.ceil(cfg.wall_thickness / cfg.horizontal_scale)), 1)
    edge_wall = int(cfg.edge_wall_height / cfg.vertical_scale)
    diagonal_wall = int(cfg.diagonal_wall_height / cfg.vertical_scale)

    heights[:wall_pixels, :] = np.maximum(heights[:wall_pixels, :], edge_wall)
    heights[-wall_pixels:, :] = np.maximum(heights[-wall_pixels:, :], edge_wall)
    heights[:, :wall_pixels] = np.maximum(heights[:, :wall_pixels], edge_wall)
    heights[:, -wall_pixels:] = np.maximum(heights[:, -wall_pixels:], edge_wall)

    rows = np.arange(width_pixels)[:, None]
    cols = np.arange(length_pixels)[None, :]
    main_diag = np.abs(rows * (length_pixels - 1) - cols * (width_pixels - 1))
    anti_diag = np.abs(
        rows * (length_pixels - 1) + cols * (width_pixels - 1) - (width_pixels - 1) * (length_pixels - 1)
    )
    diag_threshold = wall_pixels * max(width_pixels - 1, length_pixels - 1)
    diagonal_mask = (main_diag <= diag_threshold) | (anti_diag <= diag_threshold)
    heights[diagonal_mask] = heights[diagonal_mask] + diagonal_wall

    vertices, triangles = convert_height_field_to_mesh(
        heights,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    triangle_vertices = vertices[triangles]
    double_areas = np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    if np.any(double_areas <= np.finfo(vertices.dtype).eps):
        raise ValueError(
            "Stony-road terrain contains degenerate triangles; disable slope correction or adjust the height field."
        )
    # A height field only has a top surface. Close its perimeter down to a
    # buried base so the raised edge has an outward collision face even next
    # to a stair tile (whose border is at z=0).
    if cfg.base_depth <= 0:
        raise ValueError("Stony-road base_depth must be positive.")
    grid = np.arange(len(vertices)).reshape(width_pixels, length_pixels)
    perimeter = np.concatenate((grid[:, 0], grid[-1, 1:], grid[-2::-1, -1], grid[0, -2:0:-1]))
    base_vertices = vertices[perimeter].copy()
    base_vertices[:, 2] = min(0.0, float(vertices[:, 2].min())) - cfg.base_depth
    base_center = np.array([[cfg.size[0] / 2, cfg.size[1] / 2, base_vertices[0, 2]]])
    bottom = np.arange(len(perimeter)) + len(vertices)
    next_top = np.roll(perimeter, -1)
    next_bottom = np.roll(bottom, -1)
    side_faces = np.concatenate(
        (np.column_stack((perimeter, bottom, next_top)), np.column_stack((next_top, bottom, next_bottom)))
    )
    bottom_faces = np.column_stack((next_bottom, bottom, np.full(len(bottom), len(vertices) + len(bottom))))
    mesh = trimesh.Trimesh(
        vertices=np.concatenate((vertices, base_vertices, base_center)),
        faces=np.concatenate((triangles, side_faces, bottom_faces)),
    )

    x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
    x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
    y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
    y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
    origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])

    return [mesh], origin


@configclass
class HfStonyRoadTerrainCfg(HfTerrainBaseCfg):
    """Random rough terrain with low walls around each tile and along both diagonals."""

    function = stony_road_terrain

    noise_range: tuple[float, float] = MISSING
    noise_step: float = MISSING
    downsampled_scale: float | None = None
    edge_wall_height: float = 0.15
    diagonal_wall_height: float = 0.05
    wall_thickness: float = 0.05
    base_depth: float = 0.1
    """Depth of the closed base below the lowest surface height (in m)."""
