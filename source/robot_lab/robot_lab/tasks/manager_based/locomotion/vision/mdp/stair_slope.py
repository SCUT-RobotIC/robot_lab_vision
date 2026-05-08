# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg

STAIR_SLOPE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,  #若需要随机性则是设定为false，若需要确定性则设定为true
    
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.0, 0.15) ,  #障碍赛高度为100mm
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0, noise_range=(0.01, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25#障碍赛中坡度为14度，约为0.25的斜率，所以可以设定为0.4以增加难度
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
