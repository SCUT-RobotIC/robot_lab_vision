# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.low_wall import MeshConcentricLowWallTerrainCfg

from .rough_stones_env_cfg import RCRoughStonesEnvCfg


RC_CONCENTRIC_LOW_WALL_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.025,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "concentric_low_wall": MeshConcentricLowWallTerrainCfg(
            proportion=1.0,
            wall_side_lengths=(1.0, 5.0, 9.0, 13.0),
            wall_heights=(0.15, 0.20, 0.25, 0.30),
            wall_thickness=0.05,
            ground_thickness=0.02,
        ),
    },
)
"""Repeated square walls with 2 m flat corridors between wall faces."""


@configclass
class RCLowWallEnvCfg(RCRoughStonesEnvCfg):
    """RC robot low-wall crossing task initialized from rough-stones locomotion."""

    wall_half_lengths = (0.5, 2.5, 4.5, 6.5)
    wall_xs = wall_half_lengths
    wall_heights = (0.15, 0.20, 0.25, 0.30)
    wall_thickness = 0.05
    wall_max_height = 0.30
    harmless_wall_contact_name = ".*(feet|calf|calflower).*_link"
    base_body_name = "base_link"
    lidar_body_name = "lidar_link"

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = RC_CONCENTRIC_LOW_WALL_CFG
        self.scene.terrain.max_init_terrain_level = RC_CONCENTRIC_LOW_WALL_CFG.num_rows - 1
        self.scene.num_envs = 512
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.7
        self.scene.terrain.physics_material.restitution = 0.05
        # Keep reset states above the mesh floor. 0.34 m is close to the nominal
        # standing height, but with reset joint noise it can start feet inside the ground.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.38)
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 3.0
        self.terminations.terrain_out_of_bounds.params["distance_buffer"] = 0.4
        self.episode_length_s = 45.0
        
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.func = mdp.reset_root_state_cardinal_yaw
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.04, 0.08),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
            },
            "yaw_choices": (0.0, 1.57079632679, 3.14159265359, -1.57079632679),
            "yaw_jitter_range": (-0.02, 0.02),
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.02, 0.02),
                "z": (-0.04, 0.04),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
        }
        self.events.randomize_reset_joints.params["position_range"] = (-0.05, 0.05)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.1, 0.1)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-1.0, 1.0)
        self.events.randomize_push_robot.interval_range_s = (14.0, 20.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.06, 0.06),
            "roll": (-0.04, 0.04),
            "pitch": (-0.04, 0.04),
        }

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = -100.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_concentric_wall_contact_tolerant
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
            "wall_half_lengths": self.wall_half_lengths,
            "approach_distance": 0.55,
            "contact_threshold": 5.0,
            "slowdown_factor": 0.65,
        }
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.base_height_l2.weight = -4.0
        self.rewards.base_height_l2.params["target_height"] = 0.34
        self.rewards.flat_orientation_l2.weight = -0.25
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.joint_pos_limits.weight = -4.0
        self.rewards.joint_vel_limits.weight = -3.0
        self.rewards.joint_power.weight = -2.5e-5
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_mirror.func = mdp.low_wall_flat_joint_mirror
        self.rewards.joint_mirror.weight = -0.07
        self.rewards.joint_mirror.params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ],
            "wall_half_lengths": self.wall_half_lengths,
            "flat_distance": 0.55,
        }
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 90.0
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.40
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.10
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.14
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.25
        self.rewards.body_x_alignment = RewTerm(
            func=mdp.body_x_alignment_with_initial_yaw,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.low_wall_progress = RewTerm(
            func=mdp.concentric_low_wall_progress,
            weight=3.0,
            params={
                "wall_xs": self.wall_xs,
                "finish_distance": 0.9,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.low_wall_feet_clearance = RewTerm(
            func=mdp.concentric_low_wall_feet_clearance,
            weight=1.2,
            params={
                "wall_xs": self.wall_xs,
                "wall_heights": self.wall_heights,
                "approach_distance": 0.28,
                "margin": 0.06,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
            },
        )
        self.rewards.low_wall_all_feet_crossing = RewTerm(
            func=mdp.concentric_low_wall_all_feet_crossing,
            weight=2.5,
            params={
                "wall_xs": self.wall_xs,
                "completion_distance": 0.65,
                "foot_margin": 0.08,
                "lag_tolerance": 0.25,
                "positive_scale": 1.0,
                "negative_scale": 0.25,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
            },
        )
        self.rewards.low_wall_fast_crossing = RewTerm(
            func=mdp.concentric_low_wall_fast_crossing_bonus,
            weight=3.0,
            params={
                "wall_xs": self.wall_xs,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
                "contact_threshold": 5.0,
                "approach_distance": 0.35,
                "clear_distance": 0.30,
                "max_time": 4.0,
                "command_name": "base_velocity",
            },
        )
        self.rewards.low_wall_final_wall_bonus = RewTerm(
            func=mdp.concentric_low_wall_final_wall_bonus,
            weight=12.0,
            params={
                "final_wall_x": self.wall_xs[-1],
                "margin": 0.10,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
            },
        )
        self.rewards.low_wall_non_leg_contact = RewTerm(
            func=mdp.concentric_low_wall_body_contact_penalty,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        f"^(?!.*({self.harmless_wall_contact_name}|{self.base_body_name}|{self.lidar_body_name})).*"
                    ],
                ),
                "threshold": 5.0,
                "max_force": 80.0,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        f"^(?!.*({self.harmless_wall_contact_name}|{self.base_body_name}|{self.lidar_body_name})).*"
                    ],
                ),
                "wall_half_lengths": self.wall_half_lengths,
                "approach_distance": 0.45,
            },
        )
        self.rewards.low_wall_base_contact = RewTerm(
            func=mdp.concentric_low_wall_body_contact_penalty,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_body_name]),
                "threshold": 30.0,
                "max_force": 250.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_body_name]),
                "wall_half_lengths": self.wall_half_lengths,
                "approach_distance": 0.55,
            },
        )
        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        self.disable_zero_weight_rewards()


@configclass
class RCLowWallPlayEnvCfg(RCLowWallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
