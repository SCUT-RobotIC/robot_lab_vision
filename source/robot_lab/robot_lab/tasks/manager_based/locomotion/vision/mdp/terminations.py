# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
