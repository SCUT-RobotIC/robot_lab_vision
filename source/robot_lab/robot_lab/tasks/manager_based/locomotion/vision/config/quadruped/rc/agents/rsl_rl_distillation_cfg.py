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
    """学生 CNN 策略网络配置。"""

    # ========== 模型类 ==========
    class_name: str = (
        "robot_lab.tasks.manager_based.locomotion.vision.config.quadruped.rc.agents.student_cnn_policy:StudentCNNPolicy"
    )
    #class_func: type = StudentCNNPolicy

    # ========== 子模块配置 ==========
    depth_encoder_cfg: dict | None = None
    policy_head_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128)
    policy_head_activation: str = "elu"

    # ========== 观测组名 ==========
    proprio_group: str = "noise_policy"
    depth_group: str = "depth_image"

    # ========== 其他配置 ==========
    obs_normalization: bool = False
    distribution_cfg: RslRlMLPModelCfg.GaussianDistributionCfg | None = None

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
        "student": ["noise_policy"],  # 学生：本体感知 + 深度图
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
    # student = StudentCNNPolicyCfg()
    # student.depth_encoder_cfg = dict(
    #     depth_height=48,
    #     depth_width=64,
    #     output_channels=[16, 32, 32],
    #     kernel_size=[5, 4, 3],
    #     stride=[2, 2, 1],
    #     padding="same",
    #     activation="LeakyReLU",
    #     max_pool=False,
    #     global_pool="none",
    #     flatten=True,
    #     embedding_dim=128,      # 由 flat_mlp=[128] 改成 embedding_dim=128
    #     mlp_activation="elu",
    # )
    # student.policy_head_hidden_dims = [512, 256, 128]
    # student.policy_head_activation = "elu"
    # student.proprio_group = "noise_policy"
    # student.depth_group = "depth_image"
    # student.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
    #     init_std=1.0,
    # )
    # student.obs_normalization = False
    student = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )
    
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
    
