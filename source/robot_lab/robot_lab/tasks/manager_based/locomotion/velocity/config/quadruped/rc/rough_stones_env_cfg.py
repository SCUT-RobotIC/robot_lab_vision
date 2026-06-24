# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.stony_road import HfStonyRoadTerrainCfg

from .rough_env_cfg import RCRoughEnvCfg


RC_STONY_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat_stony": HfStonyRoadTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.0),
            noise_step=0.005,
            downsampled_scale=0.05,
            border_width=0.0,
        ),
        "light_stony_rough": HfStonyRoadTerrainCfg(
            proportion=0.35,
            noise_range=(0.01, 0.05),
            noise_step=0.01,
            downsampled_scale=0.05,
            border_width=0.0,
        ),
        "hard_stony_rough": HfStonyRoadTerrainCfg(
            proportion=0.25,
            noise_range=(0.05, 0.12),
            noise_step=0.02,
            downsampled_scale=0.05,
            border_width=0.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.10, 0.10),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.10, 0.10),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""Stone-like random rough terrain mix for the RC quadruped."""


@configclass
class RCRoughStonesEnvCfg(RCRoughEnvCfg):
    """RC robot training task for walking across uneven stone-like terrain."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = RC_STONY_ROAD_CFG
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.7
        self.scene.terrain.physics_material.restitution = 0.05

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.rel_standing_envs = 0.02

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.35, 1.4)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.25, 1.1)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.2)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-6.0, 6.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-4.0, 4.0)
        self.events.randomize_push_robot.interval_range_s = (10.0, 16.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "roll": (-0.12, 0.12),
            "pitch": (-0.12, 0.12),
        }

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = -100.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.lin_vel_xy_delta_l2.weight = -1.2
        self.rewards.base_height_l2.weight = -6.0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -3.0e-5
        self.rewards.joint_acc_l2.weight = -3.0e-6
        self.rewards.joint_pos_limits.weight = -4.0
        self.rewards.joint_vel_limits.weight = -3.0
        self.rewards.joint_power.weight = -2.5e-5
        self.rewards.stand_still.weight = -0.7
        self.rewards.joint_pos_penalty.weight = -0.8
        self.rewards.joint_mirror.weight = -0.04
        self.rewards.action_rate_l2.weight = -0.02

        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -2.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 80.0

        self.rewards.track_lin_vel_xy_exp.weight = 5.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5

        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.12
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -4.0
        self.rewards.feet_height_body.params["target_height"] = -0.22
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.35

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None

        self.disable_zero_weight_rewards()


@configclass
class RCRoughStonesPlayEnvCfg(RCRoughStonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
