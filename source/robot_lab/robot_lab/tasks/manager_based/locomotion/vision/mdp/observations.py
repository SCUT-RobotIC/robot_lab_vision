# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor





def feet_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """每步读取，适用于 reset 模式的摩擦随机化"""
    asset = env.scene[asset_cfg.name]
    material_props = asset.root_physx_view.get_material_properties()
    robot_friction = material_props[:, asset_cfg.body_ids, 0]
    terrain_friction = env.scene.terrain.cfg.physics_material.static_friction
    return robot_friction * terrain_friction






def terrain_step_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回当前地形的实际 step_height (m)，作为特权信息。"""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.float()
    types = terrain.terrain_types
    num_cols = terrain.cfg.num_cols
    device = env.device

    # 难度 ∈ [0, 1]
    difficulty = levels / max(num_cols - 1, 1)
    ##############################################################
    # 每种地形的 step_height_range，顺序和 sub_terrains 一致
    # (type_index, min, max)
    #############################################################
    height_configs = [
        (0, 0.0, 0.15),   # pyramid_stairs
        (1, 0.0, 0.15),   # pyramid_stairs_inv
    ]

    step_height = torch.zeros(env.num_envs, device=device)
    for type_idx, h_min, h_max in height_configs:
        mask = (types == type_idx)
        step_height[mask] = h_min + (h_max - h_min) * difficulty[mask]

    return step_height.unsqueeze(-1)  # (num_envs, 1)
        

def terrain_type_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回当前地形类型索引，作为特权信息。"""
    terrain = env.scene.terrain
    types = terrain.terrain_types.float()            # (num_envs,)
    return types.unsqueeze(-1)  