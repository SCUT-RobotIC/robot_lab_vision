# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """自适应速度命令课程学习函数
    
    根据机器人的性能表现动态调整速度命令的生成范围，实现从简单到复杂的渐进式学习。
    
    Args:
        env: 管理器基础的强化学习环境实例
        env_ids: 当前需要更新的环境ID序列
        reward_term_name: 用于评估性能的奖励项名称
        range_multiplier: 范围乘数，用于计算初始和最终速度范围，默认(0.1, 1.0)表示从10%到100%的原始范围逐步增加指令力度
    """
    # 获取基础速度命令的范围配置对象
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    
    # 仅在第一个episode初始化原始速度范围（避免重复初始化）
    if env.common_step_counter == 0:
        # 保存原始x轴速度范围到环境属性中（用于后续计算）
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        # 保存原始y轴速度范围到环境属性中
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        
        # 计算初始x轴速度范围：原始范围乘以范围乘数的第一个值（0.1 = 10%的原始范围）
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        # 计算最终x轴速度范围：原始范围乘以范围乘数的第二个值（1.0 = 100%的原始范围）
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        # 计算初始y轴速度范围
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        # 计算最终y轴速度范围
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # 将命令范围初始化为初始值（窄范围开始训练）
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # 避免每一步都更新命令课程，因为最大命令对所有环境是通用的
    # 仅在每个episode结束时更新（当全局步数整除最大episode长度时）
 
    if env.common_step_counter % env.max_episode_length == 0:
        # 获取指定奖励项的episode累计奖励（用于评估性能）
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        # 获取奖励项的配置信息（包括权重等参数）
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        # 定义命令范围的增量变化，每侧增加0.1 m/s（[-0.1, 0.1]表示范围扩大0.2 m/s）
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # 如果跟踪奖励超过最大奖励权重的80%，则增加命令范围
        # 这里计算平均奖励并归一化到每个时间步，然后与奖励权重的80%比较
        #*******************************************
        #核心
        #*******************************************
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            
            
            # 计算新的x轴速度范围：当前范围加上增量（两侧各扩大0.1 m/s）
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            # 计算新的y轴速度范围：当前范围加上增量
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # 使用clamp确保新范围不超过最终范围限制（避免超出预设的最大范围）
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # 更新命令范围配置（将张量转换回列表格式）
            #xy的指令都有更新
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    # 返回当前x轴速度范围的上限，但作为评估，是使用x进行评估
    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

