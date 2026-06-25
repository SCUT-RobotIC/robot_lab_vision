# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import RCRoughEnvCfg


@configclass
class RCFlatEnvCfg(RCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None



        # ------------------------------Sence------------------------------
        #主要不通点在于将地形进行切换。
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None#究其原因是因为训练环境是平地
        # ------------------------------Actions------------------------------
        #没有变化
        # ------------------------------Observations------------------------------
        self.observations.policy.height_scan = None
        #由于将self.scene.height_scanner = None设定为none，仍保留critic的话会报错，所以也将critic的height_scan设定为none
        self.observations.critic.height_scan = None
        # ------------------------------Events------------------------------
        #与rough相同

        # ------------------------------Rewards------------------------------
        # General
        #参数数量与rough相同，便于调整观测
        self.rewards.is_terminated.weight = -100#设定为负值则惩罚失败，可设定为-2

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_xy_delta_l2.weight = -1.0
        #若是启用则是参照当前初始状态的姿态作为‘水平’姿态
        self.rewards.base_height_l2.weight = -8.0
        self.rewards.base_height_l2.params["target_height"] = 0.30#其中障碍赛中height为300限高，即最好设定为0.2或更低
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-6
        #****************************************************************
        #这不直接设定为0而是设定成带有一定位数的负数，是为了匹配这个变量常见的数值
        #****************************************************************
        # numerical_design_wisdom = {
        #     "比例协调": {
        #         "原则": "惩罚值 ≈ 主要奖励值的1-10%",
        #         "主要奖励": "速度跟踪约2.0-3.0",
        #         "关节惩罚": "设计为0.01-0.3",
        #         "效果": "不影响学习，但能起到引导作用"
        #     },
        #     "物理直觉": {
        #         "力矩重要性": "比加速度更重要",
        #         "权重比例": "2.5e-5 vs 2.5e-7 (100倍差异)",
        #         "反映": "力矩约束比加速度约束重要100倍"
        #     }
        # }


        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = -4.0
        
        self.rewards.joint_power.weight = -2e-4
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 4.0

        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 12.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.07
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        
        
        
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_feet_link", "RR_feet_link"),
            ("FR_feet_link", "RL_feet_link"),
        )
        self.rewards.upward.weight = 0.4


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RCFlatEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Curriculums------------------------------
        # no terrain curriculum
        self.curriculum.terrain_levels = None
