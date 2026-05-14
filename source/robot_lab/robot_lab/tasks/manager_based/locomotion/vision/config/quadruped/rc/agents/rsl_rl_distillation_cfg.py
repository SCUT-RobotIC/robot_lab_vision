# Copyright (c) 2024, Your Name
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import MISSING
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlMLPModelCfg,
)
#直接导入
from student_cnn_policy import StudentCNNPolicy


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
    #class_name: str = "StudentCNNPolicy"
    class_func: type = StudentCNNPolicy  # 直接使用类对象，避免字符串解析错误
    
    # ========== 观测维度配置 ==========
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
    experiment_name: str = "rc_student_distillation"
    run_name: str = "student_cnn_policy"
    
    # ========== 观测组配置 ==========
    obs_groups: dict[str, list[str]] = {
        "student": ["noise_policy", "depth_image"],  # 学生只看本体感知 + 深度图
        "teacher": ["policy"],   # 教师额外看特权信息
    }
    
    # ========== 教师网络配置（标准 MLP）==========
    teacher: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )
    
    # ========== 学生网络配置（自定义 CNN+MLP）==========
    student: StudentCNNPolicyCfg = StudentCNNPolicyCfg(
        class_func=StudentCNNPolicy,
        
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
        
        # 策略 MLP 配置
        MLP_hidden_dims=[512, 256, 128],
        MLP_activation="elu",
        
        # 输出分布
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
        
        obs_normalization=False,
    )
    
    # ========== 蒸馏算法配置 ==========
    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        gradient_length=2,
    )
    
    # ========== 训练超参数 ==========
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100
    
