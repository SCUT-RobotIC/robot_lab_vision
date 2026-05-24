# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import RayCasterCamera, RayCaster


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
    """Return the effective foot friction after material randomization."""
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()

    # PhysX may expose the material buffer as a flattened view; reshape it to
    # (num_envs, num_shapes, 3) so we can index the selected foot bodies.
    if materials.ndim == 2:
        materials = materials.reshape(env.num_envs, -1, 3)
    elif materials.ndim == 3 and materials.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected material property shape: {tuple(materials.shape)}")

    body_ids = asset_cfg.body_ids
    if body_ids == slice(None):
        selected_materials = materials
    else:
        selected_materials = materials[:, body_ids]

    robot_static_friction = selected_materials[..., 0].to(device=env.device)

    terrain_static_friction = torch.as_tensor(
        env.scene.terrain.cfg.physics_material.static_friction, device=env.device, dtype=torch.float32
    )
    if terrain_static_friction.ndim == 0:
        terrain_static_friction = terrain_static_friction.expand_as(robot_static_friction)
    elif terrain_static_friction.ndim == 1 and terrain_static_friction.numel() == robot_static_friction.shape[-1]:
        terrain_static_friction = terrain_static_friction.unsqueeze(0).expand_as(robot_static_friction)
    else:
        terrain_static_friction = terrain_static_friction.expand_as(robot_static_friction)

    effective_friction = 0.5 * (terrain_static_friction + robot_static_friction)

    if not hasattr(env, "friction_buffer") or env.friction_buffer is None or env.friction_buffer.shape != effective_friction.shape:
        env.friction_buffer = torch.empty_like(effective_friction)

    env.friction_buffer.copy_(effective_friction)
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



def pre_pocessing_depths(
    env: ManagerBasedRLEnv,
    normalize: bool = True,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    ) -> torch.Tensor:
    """
    Computes the depth (distance) from the ray caster sensor to the hit point.
    Feed the raw information to the Encoder network
    """


    # ======================================================================
    #  Process Sensor Data
    # ======================================================================
    sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # Raw Output Shape: (num_envs, Height, Width, 1)
    depth_image = sensor.data.output["distance_to_image_plane"].clone()

    MAX_DEPTH = 3.0


    # for CNN
    # Input: (N, H, W, 1) -> Output: (N, 1, H, W)
    depth_image = depth_image.permute(0, 3, 1, 2)

    depth_image = torch.clamp(depth_image, 0.0, MAX_DEPTH)#


    

    # 假设基准误差系数为 0.005 (根据真机调参)
    # depth_image 越大的地方，噪声越大
    noise_std = 0.005 * (depth_image ** 2) 
    depth_image += torch.randn_like(depth_image) * noise_std

    Random_Artifacts_images=random_artifacts_noise(depth_image, 0.05)#考虑噪声

    pre_pocessing_depths = Random_Artifacts_images

    depth_image = pre_pocessing_depths / MAX_DEPTH  # 归一化到 [0, 1]


    return depth_image


def random_artifacts_noise(data: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    """
    对深度图进行随机伪影处理 (例如 Dropout)
    data: (num_envs,1, H, W) 或类似的形状
    """
    # 克隆一份避免原数据污染
    noisy_data = data.clone()
    
    # 例如：随机让 dropout_prob 比例的像素失效 (设为0 或 NaN，视你网络需求而定)
    mask = torch.rand_like(noisy_data) < dropout_prob
    noisy_data[mask] = 0.0 
    
    # 也可以结合高斯噪声，或者根据深度值的大小动态改变丢失率(距离越远丢失率越高)
    
    return noisy_data