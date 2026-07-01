# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.utils import configclass

try:
    from isaaclab.terrains import SubTerrainBaseCfg
except ImportError:
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def concentric_low_wall_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate light-weight concentric square walls centered at the local terrain origin."""
    center_x = 0.5 * cfg.size[0]
    center_y = 0.5 * cfg.size[1]
    meshes = [_box((cfg.size[0], cfg.size[1], cfg.ground_thickness), (center_x, center_y, -0.5 * cfg.ground_thickness))]

    for side_length, wall_height in zip(cfg.wall_side_lengths, cfg.wall_heights):
        half_side = 0.5 * side_length
        half_height = 0.5 * wall_height
        # Four independent wall segments avoid height-field tessellation and keep the map cheap to replicate.
        meshes.append(_box((cfg.wall_thickness, side_length, wall_height), (center_x + half_side, center_y, half_height)))
        meshes.append(_box((cfg.wall_thickness, side_length, wall_height), (center_x - half_side, center_y, half_height)))
        meshes.append(_box((side_length, cfg.wall_thickness, wall_height), (center_x, center_y + half_side, half_height)))
        meshes.append(_box((side_length, cfg.wall_thickness, wall_height), (center_x, center_y - half_side, half_height)))

    origin = np.array([center_x, center_y, 0.0])
    return meshes, origin


@configclass
class MeshConcentricLowWallTerrainCfg(SubTerrainBaseCfg):
    """Mesh terrain with repeated square walls around the spawn point."""

    function = concentric_low_wall_terrain

    wall_side_lengths: tuple[float, ...] = MISSING
    wall_heights: tuple[float, ...] = MISSING
    wall_thickness: float = 0.05
    ground_thickness: float = 0.02
