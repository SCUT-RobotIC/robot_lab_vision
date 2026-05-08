# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#使用 rsl_rl 库（而不是 cusrl）来进行 PPO 训练而设计的
# @configclass
# class RCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 20000
#     save_interval = 100
#     experiment_name = "rc_rough"
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0 ,
#         actor_obs_normalization=True,
#         critic_obs_normalization=True,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01 ,
#         max_grad_norm=1.0,
#     )


# @configclass
# class RCFlatPPORunnerCfg(RCRoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 5000
#         self.experiment_name = "rc_flat"



from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlMLPModelCfg,
    RslRlCNNModelCfg,
)

@configclass
class RCTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "rc_rough_teacher"
    run_name = "teacher"
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RCStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "rc_rough_student"
    run_name = "student"
    logger = "tensorboard"

    # ==================== Student: CNN + MLP ====================
    student = RslRlCNNModelCfg(
        class_name="CNNModel",
        # ----- CNN: 当前帧深度图 → 128维 embed -----
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 4, 3],
            stride=[2, 2, 1],
            padding="zeros",
            activation="elu",
            norm=["none", "none", "batch"],
            max_pool=[True, True, True],
            pool_kernel_size=[2, 2, 2],
            global_pool="none",
            flatten=True,
            output_dim=128,                # FC → 128维
        ),
        # ----- MLP: 128 + proprio × N(history) → action -----
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.01,#不需要过多的探索噪声，因为学生模型的训练完全依赖于教师模型提供的数据
        ),
    )

    # ==================== Teacher ====================
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )

    # ==================== 蒸馏算法 ====================

    #可修改为DAgger
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        gradient_length=2,
    )

    obs_groups = {
        "student": ["proprioception", "depth_image"],
        "teacher": ["proprioception", "privileged"],
    }













