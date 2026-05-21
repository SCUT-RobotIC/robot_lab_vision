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


class DepthEncoder(nn.Module):
    """CNN + flatten + embedding，用于单独部署深度图分支。"""

    def __init__(
        self,
        depth_height: int,
        depth_width: int,
        output_channels: list[int],
        kernel_size: list[int],
        stride: list[int],
        padding: str,
        activation: str,
        max_pool: bool,
        global_pool: str,
        flatten: bool,
        embedding_dim: int,
        mlp_activation: str,
    ):
        super().__init__()
        self.depth_height = depth_height
        self.depth_width = depth_width

        self.cnn_encoder = self._build_cnn_encoder(
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
            max_pool=max_pool,
            global_pool=global_pool,
            flatten=flatten,
        )

        with torch.no_grad():
            dummy_depth = torch.zeros(1, 1, depth_height, depth_width)
            cnn_output_dim = self.cnn_encoder(dummy_depth).shape[1]

        self.depth_embedding_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim, embedding_dim),
            self._get_activation(mlp_activation),
        )

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn_encoder(depth_image)
        return self.depth_embedding_mlp(cnn_features)

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
        layers = []
        in_channels = 1

        for out_ch, k_size, s in zip(output_channels, kernel_size, stride):
            if padding == "same":
                pad = k_size // 2
            else:
                pad = 0

            layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=k_size, stride=s, padding=pad)
            )
            layers.append(self._get_activation(activation))

            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_ch

        if global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))

        if flatten:
            layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
        }
        return activations.get(name.lower(), nn.ReLU())


class PolicyHead(nn.Module):
    """本体感知 + 深度 embedding -> 动作输出，可单独部署。"""

    def __init__(
        self,
        proprioception_dim: int,
        embedding_dim: int,
        num_actions: int,
        hidden_dims: list[int],
        activation: str,
        init_std: float,
        obs_normalization: bool = False,
    ):
        super().__init__()

        self.obs_normalization = obs_normalization
        self.proprioception_norm = nn.LayerNorm(proprioception_dim) if obs_normalization else nn.Identity()
        self.depth_embedding_norm = nn.LayerNorm(embedding_dim) if obs_normalization else nn.Identity()

        policy_input_dim = proprioception_dim + embedding_dim
        policy_layers = []
        input_dim = policy_input_dim

        for hidden_dim in hidden_dims:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
            ])
            input_dim = hidden_dim

        policy_layers.append(nn.Linear(input_dim, num_actions))
        self.policy_net = nn.Sequential(*policy_layers)
        self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_std)))

    def forward(
        self,
        proprioception: torch.Tensor,
        depth_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proprioception = self.proprioception_norm(proprioception)
        depth_embedding = self.depth_embedding_norm(depth_embedding)
        combined_features = torch.cat([proprioception, depth_embedding], dim=-1)
        action_mean = self.policy_net(combined_features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        return action_mean, action_std

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
        }
        return activations.get(name.lower(), nn.ReLU())


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
        obs_normalization: bool = False,
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
        flat_mlp = flat_mlp or [128]
        assert len(flat_mlp) == 1, "flat_mlp should only have one layer for embedding"
        self.embedding_dim = flat_mlp[0]

        if distribution_cfg is None:
            distribution_cfg = {"init_std": 1.0}

        init_std = distribution_cfg.get("init_std", 1.0)

        # ========== 2. 深度分支（CNN + flatten + embedding）==========
        self.depth_encoder = DepthEncoder(
            depth_height=depth_height,
            depth_width=depth_width,
            output_channels=output_channels or [16, 32, 32],
            kernel_size=kernel_size or [5, 4, 3],
            stride=stride or [2, 2, 1],
            padding=padding,
            activation=activation,
            max_pool=max_pool,
            global_pool=global_pool,
            flatten=flatten,
            embedding_dim=self.embedding_dim,
            mlp_activation=MLP_activation,
        )

        # ========== 3. 策略分支（proprioception + embedding -> action）==========
        self.policy_head = PolicyHead(
            proprioception_dim=proprioception_dim,
            embedding_dim=self.embedding_dim,
            num_actions=num_actions,
            hidden_dims=MLP_hidden_dims or [512, 256, 128],
            activation=MLP_activation,
            init_std=init_std,
            obs_normalization=obs_normalization,
        )

        # 保持旧属性名兼容训练/加载代码
        self.cnn_encoder = self.depth_encoder.cnn_encoder
        self.depth_embedding_mlp = self.depth_encoder.depth_embedding_mlp
        self.policy_net = self.policy_head.policy_net
        self.log_std = self.policy_head.log_std

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
        if not isinstance(obs, dict):
            if hasattr(obs, "shape"):
                # 例如 shape = (num_obs,)
                try:
                    return int(obs.shape[-1])
                except Exception:
                    pass
            if hasattr(obs, "num_obs"):
                return int(obs.num_obs)

        raise RuntimeError(f"Cannot infer num_obs from obs={type(obs)} for group={obs_group_name}")
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
        proprioception = observations["noise_policy"]  # (B, 93)
        depth_flat = observations["depth_image"]      # (B, 3072)
        depth_embedding = self.depth_encoder(depth_flat)
        action_mean, action_std = self.policy_head(proprioception, depth_embedding)
        
        return action_mean, action_std

    def encode_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        """单独导出深度图分支：CNN + flatten + embedding。"""
        return self.depth_encoder(depth_image)

    def policy_from_features(
        self,
        proprioception: torch.Tensor,
        depth_embedding: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """单独导出策略分支：输入 proprioception 和 embedding，输出动作。"""
        action_mean, action_std = self.policy_head(proprioception, depth_embedding)

        if deterministic:
            return action_mean

        dist = torch.distributions.Normal(action_mean, action_std)
        return dist.sample()
    
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
