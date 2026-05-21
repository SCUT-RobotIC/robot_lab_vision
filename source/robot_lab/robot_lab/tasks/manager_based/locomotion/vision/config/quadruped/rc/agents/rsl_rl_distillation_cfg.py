# Copyright (c) 2024, Your Name
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlMLPModelCfg,
)
#直接导入
from .student_cnn_policy import StudentCNNPolicy


# ========================================
# Student CNN Model Configuration
# ========================================
@configclass
class StudentCNNPolicyCfg:
    """
    学生 CNN 策略网络配置
    
    架构：
    1. Depth Image (48x64) → CNN → Flatten → MLP(128)
    2. Proprioception (93) + Depth Embedding (128) → MLP → Action
    """
    
    # ========== 模型类名 ==========
    class_name: str = "robot_lab.tasks.manager_based.locomotion.vision.config.quadruped.rc.agents.student_cnn_policy:StudentCNNPolicy"
    class_func: type = StudentCNNPolicy  # 直接使用类对象，避免字符串解析错误
    
    # ========== 观测维度配置 ==========
    obs_group_name: str = "student"  # 对应 StudentCNNPolicy 里的 obs_group_name
    proprioception_dim: int = 93  # 本体感知维度（可配置）
    depth_height: int = 48
    depth_width: int = 64
    
    # ========== CNN 网络配置 ==========
    output_channels: list[int] = MISSING  # [16, 32, 32]
    kernel_size: list[int] = MISSING      # [5, 4, 3]
    stride: list[int] = MISSING           # [2, 2, 1]
    padding: str = "zeros"                # "zeros" or "same"
    activation: str = "LeakyReLU"         # CNN 激活函数
    max_pool: bool = True                 # 是否使用 MaxPool
    global_pool: str = "none"             # "none", "avg", "max"
    flatten: bool = True                  # CNN 输出是否 flatten
    
    # ========== CNN 后的 MLP（压缩到 embedding） ==========
    flat_mlp: list[int] = MISSING         # [128] - CNN flatten 后的 MLP
    
    # ========== 策略 MLP 配置 ==========
    MLP_hidden_dims: list[int] = MISSING  # [512, 256, 128]
    MLP_activation: str = "elu"           # MLP 激活函数
    
    # ========== 输出分布配置 ==========
    distribution_cfg: RslRlMLPModelCfg.GaussianDistributionCfg = MISSING
    
    # ========== 其他配置（与 RslRlMLPModelCfg 接口对齐）==========
    obs_normalization: bool = False       # 是否归一化观测


# ========================================
# Distillation Runner Configuration
# ========================================
@configclass
class RCStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """
    Rough-terrain Climbing (RC) 学生蒸馏训练配置
    """
    
    # ========== 实验名称 ==========
    experiment_name: str = "rc_rough_teacher"
    run_name: str = "student_cnn_policy"
    
    # ========== 观测组配置 ==========
    obs_groups: dict[str, list[str]] = {
        "student": ["noise_policy", "depth_image"],  # 学生：本体感知 + 深度图
        "policy": ["noise_policy", "depth_image"],   # 兼容只认 policy 的 runner 实现
        "teacher": ["policy"],   # 教师额外看特权信息
    }
    
    # ========== 教师网络配置（标准 MLP）==========
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )
    
    # ========== 学生网络配置（自定义 CNN+MLP）==========
    student = StudentCNNPolicyCfg(
        # class_func=StudentCNNPolicy,
        class_name="robot_lab.tasks.manager_based.locomotion.vision.config.quadruped.rc.agents.student_cnn_policy:StudentCNNPolicy",
        obs_group_name="student",
        
        # CNN 配置
        output_channels=[16, 32, 32],
        kernel_size=[5, 4, 3],
        stride=[2, 2, 1],
        padding="zeros",
        activation="LeakyReLU",
        max_pool=True,
        global_pool="none",
        flatten=True,
        
        # CNN 后的压缩 MLP
        flat_mlp=[128],
        
        # 本体感知维度
        proprioception_dim=93,
        depth_height = 48,
        depth_width = 64,
        
        # 策略 MLP 配置
        MLP_hidden_dims=[512, 256, 128],
        MLP_activation="elu",
        
        # 输出分布
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
        
        obs_normalization=False,

        #obs normalization的范围？,是否需要对embedding对象进行归一化？
    )

    # 兼容不同 distillation runner 对学生侧字段命名的实现
    policy = student
    
    # ========== 蒸馏算法配置 ==========
    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        #num_mini_batches=4,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        gradient_length=2,
    )
    
    # ========== 训练超参数 ==========
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100
    
