# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def pyramid_slope_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build four planar ramps meeting a square platform, with a closed collision base."""
    width, length = cfg.size
    if not np.isclose(width, length):
        raise ValueError("Four equal-angle pyramid slopes require a square terrain tile.")
    if not 0 < cfg.slope_angle < 90:
        raise ValueError("slope_angle must be between 0 and 90 degrees.")
    if cfg.border_width < 0 or not 0 < cfg.platform_width < width - 2 * cfg.border_width:
        raise ValueError("The platform and border must leave a positive slope run.")
    if cfg.base_depth <= 0:
        raise ValueError("base_depth must be positive.")

    run = (width - 2 * cfg.border_width - cfg.platform_width) / 2
    height = run * np.tan(np.deg2rad(cfg.slope_angle))
    if cfg.inverted:
        height = -height
    center = np.array([width / 2, length / 2])

    def ring(side_length, z):
        xy = center + side_length / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        return np.column_stack((xy, np.full(4, z)))

    rings = [ring(width, 0.0)]
    if cfg.border_width > 0:
        rings.append(ring(width - 2 * cfg.border_width, 0.0))
    rings.append(ring(cfg.platform_width, height))
    faces = []
    for index in range(len(rings) - 1):
        outer, inner = 4 * index, 4 * (index + 1)
        for side in range(4):
            following = (side + 1) % 4
            faces.extend(
                ([outer + side, outer + following, inner + following], [outer + side, inner + following, inner + side])
            )
    platform = 4 * (len(rings) - 1)
    faces.extend(([platform, platform + 1, platform + 2], [platform, platform + 2, platform + 3]))

    # The inverted platform extends below zero, so its base must lie below it.
    bottom = 4 * len(rings)
    rings.append(ring(width, min(0.0, height) - cfg.base_depth))
    for side in range(4):
        following = (side + 1) % 4
        faces.extend(([side, bottom + side, following], [following, bottom + side, bottom + following]))
    faces.extend(([bottom, bottom + 2, bottom + 1], [bottom, bottom + 3, bottom + 2]))
    mesh = trimesh.Trimesh(vertices=np.concatenate(rings), faces=faces)
    return [mesh], np.array([center[0], center[1], height])


@configclass
class MeshPyramidSlopeTerrainCfg(SubTerrainBaseCfg):
    """Four constant-angle slopes rising to a square platform."""

    function = pyramid_slope_terrain
    slope_angle: float = 20.0
    """Inclination of each planar slope, in degrees."""
    platform_width: float = 1.0
    border_width: float = 1.0
    base_depth: float = 0.1
    inverted: bool = False


@configclass
class MeshInvertedPyramidSlopeTerrainCfg(MeshPyramidSlopeTerrainCfg):
    """Four constant-angle slopes descending to a square platform."""

    inverted: bool = True
