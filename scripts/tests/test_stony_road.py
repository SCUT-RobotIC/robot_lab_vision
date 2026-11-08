# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check terrain geometry and quotas: python scripts/tests/test_stony_road.py."""

import unittest

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.rc.rough_stones_env_cfg import (
    RC_STONY_ROAD_CFG as QUADRUPED_CFG,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.rc.rough_stones_env_cfg import (
    RC_STONY_ROAD_CFG as WHEELED_CFG,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.stony_road_generator import StonyRoadTerrainGenerator


class RecordingGenerator(StonyRoadTerrainGenerator):
    """Exercise the real layout generation without building hundreds of meshes."""

    def _get_terrain_mesh(self, difficulty, cfg):
        return None, difficulty

    def _add_sub_terrain(self, mesh, origin, row, col, cfg):
        self.types[row, col] = list(self.cfg.sub_terrains).index(
            next(name for name, value in self.cfg.sub_terrains.items() if value is cfg)
        )
        self.difficulties[row, col] = origin


def record_layout(cfg, rows, cols, seed, curriculum):
    generator = RecordingGenerator.__new__(RecordingGenerator)
    generator.cfg = cfg.copy()
    generator.cfg.num_rows, generator.cfg.num_cols = rows, cols
    generator.np_rng = np.random.default_rng(seed)
    generator.types = np.full((rows, cols), -1)
    generator.difficulties = np.zeros((rows, cols))
    if curriculum:
        generator._generate_curriculum_terrains()
    else:
        generator._generate_random_terrains()
    return generator.types, generator.difficulties


class TestStonyRoad(unittest.TestCase):
    def test_closed_collision_shells(self):
        for source in (QUADRUPED_CFG, WHEELED_CFG):
            for name in ("flat_stony", "light_stony_rough", "hard_stony_rough"):
                cfg = source.sub_terrains[name].copy()
                cfg.size = source.size
                cfg.horizontal_scale = source.horizontal_scale
                cfg.vertical_scale = source.vertical_scale
                cfg.slope_threshold = source.slope_threshold
                mesh = cfg.function(0.5, cfg)[0][0]
                self.assertTrue(mesh.is_watertight, name)
                self.assertTrue(mesh.is_volume, name)
                self.assertGreater(mesh.area_faces.min(), 1e-10)
                # Every exposed side must face outward and cover the whole edge,
                # including the gap from the z=0 stair border to the wall top.
                for axis in (0, 1):
                    for side, sign in ((0.0, -1), (cfg.size[axis], 1)):
                        faces = np.all(np.isclose(mesh.triangles[:, :, axis], side), axis=1)
                        self.assertTrue(np.any(faces))
                        self.assertTrue(np.all(mesh.face_normals[faces, axis] * sign > 0.99))
                        self.assertGreaterEqual(mesh.triangles[faces, :, 2].max(), cfg.edge_wall_height - 1e-6)
                        self.assertLess(mesh.triangles[faces, :, 2].min(), 0)

    def test_planar_pyramid_slopes(self):
        for source in (QUADRUPED_CFG, WHEELED_CFG):
            for name, sign in (("pyramid_slope", 1), ("pyramid_slope_inv", -1)):
                cfg = source.sub_terrains[name].copy()
                cfg.size = source.size
                for difficulty in (0.0, 0.5, 1.0):
                    meshes, origin = cfg.function(difficulty, cfg)
                    mesh = meshes[0]
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_volume)
                    self.assertGreater(mesh.area_faces.min(), 1e-10)
                    slope_faces = (mesh.face_normals[:, 2] > 0) & (mesh.face_normals[:, 2] < 1 - 1e-8)
                    self.assertEqual(slope_faces.sum(), 8)
                    angles = np.rad2deg(np.arccos(mesh.face_normals[slope_faces, 2]))
                    np.testing.assert_allclose(angles, 20.0, atol=1e-8)
                    # Two triangles per direction, with no cross slope or curved surface.
                    normals = mesh.face_normals[slope_faces, :2]
                    self.assertEqual(len(np.unique(np.round(normals, 8), axis=0)), 4)
                    self.assertTrue(np.all(np.isclose(normals, 0).sum(axis=1) == 1))
                    height = sign * 2.5 * np.tan(np.deg2rad(20.0))
                    np.testing.assert_allclose(origin, [4.0, 4.0, height])
                    platform = np.all(np.isclose(mesh.triangles[:, :, 2], height), axis=1)
                    self.assertAlmostEqual(mesh.area_faces[platform].sum(), 1.0)
                    platform_vertices = mesh.triangles[platform].reshape(-1, 3)
                    np.testing.assert_allclose(np.ptp(platform_vertices[:, :2], axis=0), [1.0, 1.0])
                    self.assertLess(mesh.bounds[0, 2], min(0.0, height))

    def test_terrain_quotas_and_curriculum(self):
        for source in (QUADRUPED_CFG, WHEELED_CFG):
            self.assertIs(source.class_type, StonyRoadTerrainGenerator)
            for rows, cols in ((5, 5), (10, 10), (10, 20)):
                for curriculum in (False, True):
                    for seed in range(20):
                        types, difficulties = record_layout(source, rows, cols, seed, curriculum)
                        counts = np.bincount(types.ravel(), minlength=len(source.sub_terrains))
                        for name, minimum in source.minimum_proportions.items():
                            index = list(source.sub_terrains).index(name)
                            self.assertGreaterEqual(counts[index], int(np.ceil(rows * cols * minimum)))
                        self.assertEqual(counts.sum(), rows * cols)
                        if (rows, cols) == (10, 10) and not curriculum:
                            np.testing.assert_array_equal(counts, [25, 30, 15, 10, 10, 5, 5])
                        if (rows, cols) == (5, 5) and not curriculum:
                            np.testing.assert_array_equal(counts, [6, 8, 4, 3, 2, 1, 1])
                        if curriculum:
                            np.testing.assert_array_equal(types, np.tile(types[0], (rows, 1)))
                            self.assertTrue(np.all(np.diff(difficulties, axis=0) > 0))
                        self.assertTrue(np.all((difficulties >= 0) & (difficulties <= 1)))
            first = record_layout(source, 10, 10, 42, False)[0]
            np.testing.assert_array_equal(first, record_layout(source, 10, 10, 42, False)[0])
            self.assertFalse(np.array_equal(first, record_layout(source, 10, 10, 43, False)[0]))

    def test_small_layout_rejects_impossible_quotas(self):
        with self.assertRaisesRegex(ValueError, "Too few terrain slots"):
            record_layout(WHEELED_CFG, 1, 1, 0, False)

    def test_complete_generator(self):
        cfg = WHEELED_CFG.copy()
        cfg.num_rows = cfg.num_cols = 5
        cfg.seed = 42
        generator = cfg.class_type(cfg, device="cpu")
        self.assertEqual(generator.terrain_origins.shape, (5, 5, 3))
        self.assertTrue(np.all(np.isfinite(generator.terrain_mesh.vertices)))
        self.assertGreater(generator.terrain_mesh.area_faces.min(), 1e-10)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        app.close()
