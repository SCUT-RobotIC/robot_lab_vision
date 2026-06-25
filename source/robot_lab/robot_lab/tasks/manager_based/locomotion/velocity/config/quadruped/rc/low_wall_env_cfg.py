# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_stones_env_cfg import RCRoughStonesEnvCfg


@configclass
class RCLowWallEnvCfg(RCRoughStonesEnvCfg):
    """RC robot low-wall crossing task initialized from rough-stones locomotion."""

    wall_x = 1.2
    wall_width = 1.2
    wall_thickness = 0.05
    wall_max_height = 0.30
    wall_height_range = (0.15, 0.30)
    harmless_wall_contact_name = ".*(feet|calf|calflower).*_link"
    fragile_body_name = "base_link|lidar_link"

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.7
        self.scene.terrain.physics_material.restitution = 0.05
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.34)

        self.scene.low_wall = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/LowWall",
            spawn=sim_utils.CuboidCfg(
                size=(self.wall_thickness, self.wall_width, self.wall_max_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.02,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.30)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.wall_x, 0.0, 0.5 * self.wall_height_range[0])),
        )
        self.scene.low_wall_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/LowWall",
            history_length=3,
            track_air_time=False,
        )

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0

        # ------------------------------Events------------------------------
        self.events.randomize_low_wall = EventTerm(
            func=mdp.reset_low_wall_position,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("low_wall"),
                "wall_x": self.wall_x,
                "wall_height_range": self.wall_height_range,
                "wall_max_height": self.wall_max_height,
            },
        )
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.12, 0.02),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.04),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.02, 0.02),
            },
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.02, 0.02),
                "z": (-0.04, 0.04),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
        }
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
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_wall_contact_tolerant
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[f"^(?!.*{self.foot_link_name}).*"]),
            "wall_x": self.wall_x,
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
            func=mdp.body_x_alignment,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.low_wall_progress = RewTerm(
            func=mdp.low_wall_progress,
            weight=2.0,
            params={
                "wall_x": self.wall_x,
                "finish_distance": 0.9,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.low_wall_feet_clearance = RewTerm(
            func=mdp.low_wall_feet_clearance,
            weight=1.2,
            params={
                "wall_x": self.wall_x,
                "approach_distance": 0.28,
                "margin": 0.06,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "command_name": "base_velocity",
            },
        )
        self.rewards.low_wall_non_leg_contact = RewTerm(
            func=mdp.low_wall_body_contact_penalty,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[f"^(?!.*({self.harmless_wall_contact_name}|{self.fragile_body_name})).*"],
                ),
                "threshold": 5.0,
                "max_force": 80.0,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[f"^(?!.*({self.harmless_wall_contact_name}|{self.fragile_body_name})).*"],
                ),
                "wall_x": self.wall_x,
                "approach_distance": 0.45,
            },
        )
        self.rewards.low_wall_base_lidar_impact = RewTerm(
            func=mdp.low_wall_body_contact_penalty,
            weight=-8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.fragile_body_name]),
                "threshold": 20.0,
                "max_force": 120.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.fragile_body_name]),
                "wall_x": self.wall_x,
                "approach_distance": 0.55,
            },
        )
        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.low_wall_hard_base_lidar_impact = DoneTerm(
            func=mdp.selected_body_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.fragile_body_name]),
                "threshold": 220.0,
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.low_wall_height = CurrTerm(
            func=mdp.low_wall_height_curriculum,
            params={
                "reward_term_name": "low_wall_progress",
                "success_threshold": 0.65,
                "increment": 0.05,
            },
        )

        self.disable_zero_weight_rewards()


@configclass
class RCLowWallPlayEnvCfg(RCLowWallEnvCfg):
    wall_height_range = (0.30, 0.30)

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.curriculum.low_wall_height = None
