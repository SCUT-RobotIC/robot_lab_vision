# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.utils import configclass


class StonyRoadTerrainGenerator(TerrainGenerator):
    """Allocate terrain counts before shuffling, reserving stairs and slopes."""

    def _get_sub_terrain_indices(self, count: int) -> np.ndarray:
        names = list(self.cfg.sub_terrains)
        weights = np.array([cfg.proportion for cfg in self.cfg.sub_terrains.values()], dtype=float)
        if np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            raise ValueError("Terrain proportions must be finite, non-negative and have a positive sum.")
        targets = count * weights / weights.sum()
        minimums = np.zeros(len(names), dtype=int)
        for name, proportion in self.cfg.minimum_proportions.items():
            if not 0 <= proportion <= 1:
                raise ValueError("Minimum terrain proportions must be between zero and one.")
            minimums[names.index(name)] = int(np.ceil(count * proportion))
        if minimums.sum() > count:
            raise ValueError("Too few terrain slots to satisfy the minimum terrain proportions.")
        counts = np.maximum(np.floor(targets).astype(int), minimums)
        # Largest-remainder rounding, respecting the reserved terrain slots.
        while counts.sum() > count:
            surplus = np.where(counts > minimums, counts - targets, -np.inf)
            counts[np.argmax(surplus)] -= 1
        while counts.sum() < count:
            counts[np.argmax(targets - counts)] += 1
        return np.repeat(np.arange(len(names)), counts)

    def _generate_random_terrains(self):
        self._generate_stony_road_terrains(curriculum=False)

    def _generate_curriculum_terrains(self):
        self._generate_stony_road_terrains(curriculum=True)

    def _generate_stony_road_terrains(self, curriculum: bool):
        configs = list(self.cfg.sub_terrains.values())
        if curriculum:
            # Keep one type per column so terrain-level progression still works.
            indices = self._get_sub_terrain_indices(self.cfg.num_cols)
            indices = np.tile(indices, (self.cfg.num_rows, 1))
        else:
            indices = self._get_sub_terrain_indices(self.cfg.num_rows * self.cfg.num_cols)
            self.np_rng.shuffle(indices)
            indices = indices.reshape(self.cfg.num_rows, self.cfg.num_cols)
        for row, col in np.ndindex(indices.shape):
            if curriculum:
                lower, upper = self.cfg.difficulty_range
                difficulty = lower + (upper - lower) * (row + self.np_rng.uniform()) / self.cfg.num_rows
            else:
                difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            cfg = configs[indices[row, col]]
            mesh, origin = self._get_terrain_mesh(difficulty, cfg)
            self._add_sub_terrain(mesh, origin, row, col, cfg)


@configclass
class StonyRoadTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Stony-road mix reserving at least 5% of each stair and 2.5% of each slope."""

    class_type: type = StonyRoadTerrainGenerator
    minimum_proportions: dict[str, float] = {
        "pyramid_stairs": 0.05,
        "pyramid_stairs_inv": 0.05,
        "pyramid_slope": 0.025,
        "pyramid_slope_inv": 0.025,
    }
