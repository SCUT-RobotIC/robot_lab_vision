# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dead_down(
    env: ManagerBasedRLEnv, threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当机器人翻倒时终止回合。
    
    通过检测 projected_gravity_b[:, 2] 来判断机器人是否翻倒：
    - 机器人直立时：projected_gravity_b[:, 2] ≈ -1.0
    - 机器人倒立时：projected_gravity_b[:, 2] ≈ +1.0
    - 当该值大于 threshold 时，认为机器人已翻倒
    
    Args:
        env: 环境实例
        threshold: 判定翻倒的阈值，默认 0.5（对应约 120° 倾斜）
        asset_cfg: 资产配置
        
    Returns:
        布尔张量，指示哪些环境应该终止
    """
    # 提取资产数据
    asset: RigidObject = env.scene[asset_cfg.name]
    # 当 projected_gravity_b[:, 2] > threshold 时，机器人已经翻倒
    return asset.data.projected_gravity_b[:, 2] > threshold


def low_bar_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("low_bar_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when the low bar receives any meaningful contact force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold, dim=1)


def selected_body_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Terminate when selected bodies receive an excessive contact force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.any(torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold, dim=1)


def broken_bridge_groove_bottom_contacts(
    env: ManagerBasedRLEnv,
    x_flat_length: float,
    groove_width: float,
    groove_depth: float,
    skip_every_n_groove: int,
    early_contact_count: int,
    late_contact_count: int,
    switch_step: int,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    bottom_margin: float = 0.03,
) -> torch.Tensor:
    """Terminate when too many feet contact the bottom of active grooves."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    local_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - env.scene.env_origins[:, None, :]
    x = local_pos[:, :, 0]
    z = local_pos[:, :, 2]
    x_cycle_length = x_flat_length + groove_width
    x_phase = torch.remainder(x + 0.5 * x_cycle_length, x_cycle_length)
    x_groove_index = torch.floor((x + 0.5 * x_cycle_length) / x_cycle_length).long()
    in_groove = x_phase >= x_flat_length
    if skip_every_n_groove > 0:
        in_groove &= torch.remainder(x_groove_index + 1, skip_every_n_groove) != 0
    near_bottom = in_groove & (z <= -groove_depth + bottom_margin)

    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    bottom_contacts = near_bottom & (contact_force > contact_threshold)
    contact_count = torch.sum(bottom_contacts, dim=1)
    threshold = early_contact_count if env.common_step_counter < switch_step else late_contact_count
    return contact_count >= threshold
