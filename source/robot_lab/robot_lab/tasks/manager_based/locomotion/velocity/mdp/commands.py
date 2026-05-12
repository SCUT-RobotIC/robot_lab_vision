# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        #对速度命令进行阈值处理，将过小的线速度命令设为零（此处为0.2 m/s）

        # 张量形状: (num_envs, 3) = (4096, 3)
        # 其中: [v_x, v_y, v_theta] 或 [v_x, v_y, heading]
            # env_ids: 环境ID序列，如 [0, 1, 2, 10, 11]
            # :2: 选择前两列（线速度分量）

        #torch.norm(self.vel_command_b[env_ids, :2], dim=1)
        # input: 选定环境的线速度张量，形状 (len(env_ids), 2)
        # p=2: 默认L2范数（欧几里得距离）
        # dim=1: 在维度1上计算范数（跨列计算）
        # 计算过程：
        # 对于每个环境: norm = √(v_x² + v_y²)
        # 结果形状: (len(env_ids),) - 每个环境的线速度模长

        #.unsqueeze(1)
        # 输入: 布尔张量，形状 (len(env_ids),)
        # dim=1: 在维度1插入新维度
        # 输出: 形状 (len(env_ids), 1)
        # 目的: 为广播操作准备维度
            #系统会自动扩展较小的张量来匹配较大张量的形状。

        #应用掩码（原地修改）
        #line_velocities *= mask_expanded  # 形状保持: (N, 2)
        # True → 乘以1（保持），False → 乘以0（清零）




@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""
    #更适用于状态机任务，可将障碍任务进行分割
    class_type: type = UniformThresholdVelocityCommand
    #创建与自定义命令生成器对应的配置类


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    #指令是预先设定好的，从预定义的离散值列表中随机选择命令
    #适用于分类任务、状态切换
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg#用于指定cfg的类型
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers所有命令值都是整数类型
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        #确认设备和创建命令缓冲区
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:#确定字符串表达方式，可用于显示环境数量和可用命令列表
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        #
        #留下接口可供扩展
        #
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""#为指定环境重新采样命令
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
