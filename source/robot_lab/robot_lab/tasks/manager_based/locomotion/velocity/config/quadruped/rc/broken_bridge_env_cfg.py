# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.broken_bridge import HfBrokenBridgeTerrainCfg

from .rough_stones_env_cfg import RCRoughStonesEnvCfg


RC_BROKEN_BRIDGE_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "broken_bridge": HfBrokenBridgeTerrainCfg(
            proportion=1.0,
            x_flat_length=0.40,
            groove_width=0.15,
            groove_depth=0.20,
            skip_every_n_groove=7,
            border_width=0.0,
        ),
    },
)
"""Flat field with 0.4 m flat spans separated by 0.15 m wide x-direction grooves."""


@configclass
class RCBrokenBridgeEnvCfg(RCRoughStonesEnvCfg):
    """RC robot broken-bridge task initialized from rough-stones locomotion."""

    x_flat_length = 0.40
    groove_width = 0.15
    groove_depth = 0.20
    skip_every_n_groove = 7
    start_x = -0.20
    finish_x = 3.5
    termination_switch_step = 400_000

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = RC_BROKEN_BRIDGE_CFG
        self.scene.terrain.physics_material.static_friction = 0.9
        self.scene.terrain.physics_material.dynamic_friction = 0.8
        self.scene.terrain.physics_material.restitution = 0.02
        self.scene.robot.init_state.pos = (self.start_x, 0.0, 0.34)

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.08, 0.08),
                "z": (0.0, 0.04),
                "roll": (-0.04, 0.04),
                "pitch": (-0.04, 0.04),
                "yaw": (-0.08, 0.08),
            },
            "velocity_range": {
                "x": (-0.04, 0.04),
                "y": (-0.03, 0.03),
                "z": (-0.04, 0.04),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.04, 0.04),
            },
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.55, 1.4)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 1.1)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-2.5, 2.5)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-1.5, 1.5)
        self.events.randomize_push_robot.interval_range_s = (14.0, 20.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.08, 0.08),
            "roll": (-0.04, 0.04),
            "pitch": (-0.04, 0.04),
        }

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = -100.0
        self.rewards.track_lin_vel_xy_exp.weight = 4.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.base_height_l2.weight = -4.0
        self.rewards.base_height_l2.params["target_height"] = 0.34
        self.rewards.flat_orientation_l2.weight = -0.25
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.joint_pos_limits.weight = -4.0
        self.rewards.joint_vel_limits.weight = -3.0
        self.rewards.joint_power.weight = -2.5e-5
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 90.0
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.12
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.14
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.3

        groove_params = {
            "x_flat_length": self.x_flat_length,
            "groove_width": self.groove_width,
            "groove_depth": self.groove_depth,
            "skip_every_n_groove": self.skip_every_n_groove,
        }
        self.rewards.body_x_alignment = RewTerm(
            func=mdp.body_x_alignment,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.broken_bridge_progress = RewTerm(
            func=mdp.broken_bridge_progress,
            weight=2.0,
            params={
                "start_x": self.start_x,
                "finish_x": self.finish_x,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.broken_bridge_centerline = RewTerm(
            func=mdp.broken_bridge_centerline,
            weight=1.0,
            params={
                "std": 0.65,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.broken_bridge_feet_on_blocks = RewTerm(
            func=mdp.broken_bridge_feet_on_blocks,
            weight=2.0,
            params={
                **groove_params,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
            },
        )
        self.rewards.broken_bridge_feet_above_groove = RewTerm(
            func=mdp.broken_bridge_feet_above_groove,
            weight=-0.6,
            params={
                **groove_params,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.broken_bridge_feet_recovery = RewTerm(
            func=mdp.BrokenBridgeFeetRecoveryReward,
            weight=0.75,
            params={
                **groove_params,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.broken_bridge_feet_bottom_contact = RewTerm(
            func=mdp.broken_bridge_feet_bottom_contact,
            weight=-8.0,
            params={
                **groove_params,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.broken_bridge_bottom_contacts = DoneTerm(
            func=mdp.broken_bridge_groove_bottom_contacts,
            params={
                **groove_params,
                "early_contact_count": 3,
                "late_contact_count": 2,
                "switch_step": self.termination_switch_step,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "contact_threshold": 1.0,
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        self.disable_zero_weight_rewards()


@configclass
class RCBrokenBridgePlayEnvCfg(RCBrokenBridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
