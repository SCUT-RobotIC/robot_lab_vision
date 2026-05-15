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
    # 初始化缓冲区：形状为 (num_envs, 4)，初始值为 0.3
    if not hasattr(env, "friction_buffer") or env.friction_buffer is None:
        env.friction_buffer = torch.full(
            (env.num_envs, 4), 0.3, device=env.device, dtype=torch.float32
        )
    
    # 从环境中读取 terrain_friction
    terrain_friction = env.scene.terrain.cfg.physics_material.static_friction
    terrain_friction = torch.as_tensor(
        terrain_friction, device=env.device, dtype=torch.float32
    )
    
    # 用 terrain_friction 覆盖缓冲区的值
    env.friction_buffer.fill_(terrain_friction)
    
    return env.friction_buffer



def terrain_step_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回当前地形的实际 step_height (m)，作为特权信息。"""
    from .utils import _get_terrain_column_range

    terrain = env.scene.terrain
    terrain_cfg = getattr(terrain.cfg, "terrain_generator", None)
    device = env.device
    
    step_height = torch.zeros(env.num_envs, device=device)
    if terrain_cfg is None or terrain_cfg.sub_terrains is None:
        return step_height.unsqueeze(-1)
        
    levels = terrain.terrain_levels.float()
    types = terrain.terrain_types  # 这个在 IsaacLab 是列索引(column index)
    
    difficulty = levels / max(terrain_cfg.num_rows - 1, 1)

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())

    for terrain_name in sub_terrain_names:
        sub_cfg = terrain_cfg.sub_terrains[terrain_name]
        
        h_min, h_max = 0.0, 0.0
        if hasattr(sub_cfg, "step_height_range"):
            h_min, h_max = sub_cfg.step_height_range
            
        if h_min != 0.0 or h_max != 0.0:
            # 💡 直接调用你的内置工具函数
            col_range = _get_terrain_column_range(terrain_cfg, terrain_name, device)
            if col_range is not None:
                col_start, col_end = col_range
                mask = (types >= col_start) & (types < col_end)
                step_height[mask] = h_min + (h_max - h_min) * difficulty[mask]

    return step_height.unsqueeze(-1)  # (num_envs, 1)
        

def terrain_type_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回当前地形类型索引，作为特权信息。"""
    terrain = env.scene.terrain
    types = torch.as_tensor(terrain.terrain_types, device=env.device, dtype=torch.float32)
    return types.unsqueeze(-1)