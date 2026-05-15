# Copyright (c) 2024, Your Name
# SPDX-License-Identifier: MIT

"""
Student CNN Policy Network for Distillation

架构：
    1. Depth Image (1, 48, 64) → CNN Encoder → Flatten
    2. CNN Flatten → Single Layer MLP → Depth Embedding (128)
    3. Proprioception (93) + Depth Embedding (128) → Policy MLP → Action
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple


class StudentCNNPolicy(nn.Module):
    """
    学生 CNN 策略网络（用于蒸馏训练）
    
    只实现 Policy 网络（无 Critic，因为蒸馏不需要价值函数）
    """
    
    def __init__(
        self,
        obs,                 # rsl_rl 传入的 obs 描述（不是 int）
        obs_groups,          # dict: {"student":[...], "teacher":[...]}
        obs_group_name: str, # "student"
        num_actions: int,
        proprioception_dim: int = 93,
        depth_height: int = 48,
        depth_width: int = 64,
        output_channels: list[int] | None = None,
        kernel_size: list[int] | None = None,
        stride: list[int] | None = None,
        padding: str = "zeros",
        activation: str = "LeakyReLU",
        max_pool: bool = True,
        global_pool: str = "none",
        flatten: bool = True,
        flat_mlp: list[int] | None = None,
        MLP_hidden_dims: list[int] | None = None,
        MLP_activation: str = "elu",
        distribution_cfg=None,
        **kwargs,
    ):
        super().__init__()

        # 关键：从 obs 里拿到这个 group 的观测维度
        # 不同版本 rsl_rl 的 obs 结构不一样，这里做个兼容写法：
        num_obs = self._infer_num_obs(obs, obs_groups, obs_group_name)

        self.proprioception_dim = proprioception_dim
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_dim = depth_height * depth_width
        self.num_actions = num_actions

        expected_obs_dim = proprioception_dim + self.depth_dim
        assert num_obs == expected_obs_dim, (
            f"Observation dimension mismatch! Expected {expected_obs_dim}, got {num_obs}."
        )
        
        # ========== 2. 构建 CNN 编码器 ==========
        self.cnn_encoder = self._build_cnn_encoder(
            output_channels=output_channels or [16, 32, 32],
            kernel_size=kernel_size or [5, 4, 3],
            stride=stride or [2, 2, 1],
            padding=padding,
            activation=activation,
            max_pool=max_pool,
            global_pool=global_pool,
            flatten=flatten,
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            dummy_depth = torch.zeros(1, 1, depth_height, depth_width)
            cnn_output_dim = self.cnn_encoder(dummy_depth).shape[1]
        
        # ========== 3. CNN Flatten → MLP (压缩到 embedding) ==========
        flat_mlp = flat_mlp or [128]
        assert len(flat_mlp) == 1, "flat_mlp should only have one layer for embedding"
        self.embedding_dim = flat_mlp[0]
        
        mlp_activation = self._get_activation(MLP_activation)
        self.depth_embedding_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim, self.embedding_dim),
            mlp_activation
        )
        
        # ========== 4. 策略 MLP（拼接特征 → 动作均值）==========
        policy_input_dim = proprioception_dim + self.embedding_dim  # 93 + 128 = 221
        MLP_hidden_dims = MLP_hidden_dims or [512, 256, 128]
        
        policy_layers = []
        input_dim = policy_input_dim
        for hidden_dim in MLP_hidden_dims:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(MLP_activation)
            ])
            input_dim = hidden_dim
        
        # 输出层：动作均值
        policy_layers.append(nn.Linear(input_dim, num_actions))
        self.policy_net = nn.Sequential(*policy_layers)
        
        # ========== 5. 高斯分布配置（标准差）==========
        if distribution_cfg is None:
            distribution_cfg = {"init_std": 1.0}
        
        init_std = distribution_cfg.get("init_std", 1.0)
        self.log_std = nn.Parameter(
            torch.ones(num_actions) * torch.log(torch.tensor(init_std))
        )
        
        # # ========== 6. 打印网络架构信息 ==========
        # print("\n" + "="*80)
        # print("🎓 Student CNN Policy Architecture")
        # print("="*80)
        # print(f"📊 Observation Dimensions:")
        # print(f"   - Proprioception: {proprioception_dim}")
        # print(f"   - Depth Image: ({depth_height}, {depth_width}) = {self.depth_dim} dims")
        # print(f"   - Total: {num_obs}")
        # print(f"\n🔍 CNN Encoder:")
        # print(f"   - Input: (1, {depth_height}, {depth_width})")
        # print(f"   - Output (after flatten): {cnn_output_dim} dims")
        # print(f"\n🧠 Depth Embedding MLP:")
        # print(f"   - {cnn_output_dim} → {self.embedding_dim}")
        # print(f"\n🎯 Policy MLP:")
        # print(f"   - Input: {policy_input_dim} (proprio + depth embedding)")
        # print(f"   - Hidden: {MLP_hidden_dims}")
        # print(f"   - Output: {num_actions} actions")
        # print(f"\n📉 Distribution: Gaussian (init_std={init_std:.2f})")
        # print("="*80 + "\n")
    def _infer_num_obs(self, obs, obs_groups, obs_group_name: str) -> int:
        # 下面给几个常见情况的兜底，你根据你实际 obs 结构选一个成立的
        # 1) 如果 obs 直接就是 int
        if isinstance(obs, int):
            return obs

        # 2) 如果 obs 是 dict，里面按 group_name 存维度
        if isinstance(obs, dict):
            if obs_group_name in obs and isinstance(obs[obs_group_name], int):
                return obs[obs_group_name]
            if obs_group_name in obs and hasattr(obs[obs_group_name], "__len__"):
                return len(obs[obs_group_name])

        # 3) 如果 obs 有 num_obs 或 shape 等属性（常见是 tensor-like spec）
        if hasattr(obs, "shape"):
            # 例如 shape = (num_obs,)
            try:
                return int(obs.shape[-1])
            except Exception:
                pass
        if hasattr(obs, "num_obs"):
            return int(obs.num_obs)

        raise RuntimeError(f"Cannot infer num_obs from obs={type(obs)} for group={obs_group_name}")
    def _build_cnn_encoder(
        self,
        output_channels: list[int],
        kernel_size: list[int],
        stride: list[int],
        padding: str,
        activation: str,
        max_pool: bool,
        global_pool: str,
        flatten: bool,
    ) -> nn.Module:
        """
        构建 CNN 编码器
        
        Args:
            output_channels: 每层输出通道数 [16, 32, 32]
            kernel_size: 每层卷积核大小 [5, 4, 3]
            stride: 每层步幅 [2, 2, 1]
            padding: "zeros" or "same"
            activation: 激活函数名称
            max_pool: 是否使用 MaxPool
            global_pool: "none", "avg", "max"
            flatten: 是否最后 flatten
        """
        layers = []
        in_channels = 1  # 单通道深度图
        
        # 构建卷积层
        for out_ch, k_size, s in zip(output_channels, kernel_size, stride):
            # 计算 padding
            if padding == "same":
                pad = k_size // 2
            else:
                pad = 0
            
            # 卷积层
            layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=k_size, stride=s, padding=pad)
            )
            
            # 激活函数
            layers.append(self._get_activation(activation))
            
            # MaxPool（可选）
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_ch
        
        # 全局池化（可选）
        if global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        
        # Flatten
        if flatten:
            layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            observations: (batch_size, num_obs)
                - [:, :proprioception_dim]: 本体感知
                - [:, proprioception_dim:]: 深度图 flatten
        
        Returns:
            action_mean: (batch_size, num_actions)
            action_std: (batch_size, num_actions)
        """
        batch_size = observations.shape[0]
        
        # ========== 1. 分割观测 ==========
        proprioception = observations[:, :self.proprioception_dim]  # (B, 93)
        depth_flat = observations[:, self.proprioception_dim:]      # (B, 3072)
        
        # ========== 2. Reshape 深度图 ==========
        depth_image = depth_flat.view(
            batch_size, 1, self.depth_height, self.depth_width
        )  # (B, 1, 48, 64)
        
        # ========== 3. CNN 编码 ==========
        cnn_features = self.cnn_encoder(depth_image)  # (B, cnn_output_dim)
        
        # ========== 4. 深度 Embedding MLP ==========
        depth_embedding = self.depth_embedding_mlp(cnn_features)  # (B, 128)
        
        # ========== 5. 拼接特征 ==========
        combined_features = torch.cat(
            [proprioception, depth_embedding], dim=-1
        )  # (B, 93+128=221)
        
        # ========== 6. 策略网络输出动作均值 ==========
        action_mean = self.policy_net(combined_features)  # (B, num_actions)
        
        # ========== 7. 动作标准差（可学习）==========
        action_std = torch.exp(self.log_std).expand_as(action_mean)  # (B, num_actions)
        
        return action_mean, action_std
    
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        推理时使用的动作选择
        
        Args:
            observations: (batch_size, num_obs)
            deterministic: 是否使用确定性策略（取均值）
        
        Returns:
            actions: (batch_size, num_actions)
        """
        action_mean, action_std = self.forward(observations)
        
        if deterministic:
            return action_mean
        else:
            # 从高斯分布采样
            dist = torch.distributions.Normal(action_mean, action_std)
            return dist.sample()
    
    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率（用于蒸馏训练）
        
        Args:
            observations: (batch_size, num_obs)
            actions: (batch_size, num_actions)
        
        Returns:
            log_prob: (batch_size,)
            entropy: (batch_size,)
        """
        action_mean, action_std = self.forward(observations)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def reset(self, env_ids: torch.Tensor = None):
        """重置网络状态（如果有 RNN 等需要）"""
        pass
