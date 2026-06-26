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


def broken_bridge_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat field cut by periodic x/y grooves."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1

    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    groove_depth = int(round(cfg.groove_depth / cfg.vertical_scale))

    x_coords = np.arange(width_pixels, dtype=np.float32) * cfg.horizontal_scale - 0.5 * cfg.size[0]

    x_cycle_length = cfg.x_flat_length + cfg.groove_width
    x_phase = np.mod(x_coords + 0.5 * x_cycle_length, x_cycle_length)
    x_groove_index = np.floor((x_coords + 0.5 * x_cycle_length) / x_cycle_length).astype(np.int64)
    x_groove = x_phase >= cfg.x_flat_length
    if cfg.skip_every_n_groove > 0:
        x_groove &= np.mod(x_groove_index + 1, cfg.skip_every_n_groove) != 0

    groove_mask = x_groove[:, None] & np.ones((1, length_pixels), dtype=bool)
    heights[groove_mask] = -groove_depth

    vertices, triangles = convert_height_field_to_mesh(
        heights,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    origin_x = 0.5 * cfg.size[0]
    origin_y = 0.5 * cfg.size[1]
    origin = np.array([origin_x, origin_y, 0.0])

    return [mesh], origin


@configclass
class HfBrokenBridgeTerrainCfg(HfTerrainBaseCfg):
    """Height-field terrain made of a flat field with periodic grooves."""

    function = broken_bridge_terrain

    x_flat_length: float = MISSING
    groove_width: float = MISSING
    groove_depth: float = MISSING
    skip_every_n_groove: int = 0
