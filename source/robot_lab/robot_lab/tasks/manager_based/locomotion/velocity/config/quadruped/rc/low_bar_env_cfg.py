# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import RCRoughEnvCfg


LOW_BAR_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class RCLowBarEnvCfg(RCRoughEnvCfg):
    """RC robot low-bar crossing task.

    The policy receives a scalar base-height command and must pass under a
    fragile low bar without touching it.
    """

    base_link_name = "base_link"
    foot_link_name = ".*_feet_joint"

    low_bar_x = 1.0
    low_bar_height = 0.30
    low_bar_radius = 0.015
    low_bar_width = 1.2

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = LOW_BAR_FLAT_TERRAIN_CFG

        self.scene.low_bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/LowBar",
            spawn=sim_utils.CylinderCfg(
                radius=self.low_bar_radius,
                height=self.low_bar_width,
                axis="Y",
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.08)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.low_bar_x, 0.0, self.low_bar_height)),
        )
        self.scene.low_bar_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/LowBar",
            history_length=3,
            track_air_time=False,
        )

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.15, 0.15)
        self.commands.base_velocity.ranges.heading = (-0.1, 0.1)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_height = mdp.UniformBaseHeightCommandCfg(
            resampling_time_range=(4.0, 8.0),
            height_range=(0.20, 0.28),
        )

        # ------------------------------Observations------------------------------
        self.observations.policy.base_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height"},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        self.observations.critic.base_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height"},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.15, 0.05),
                "y": (-0.08, 0.08),
                "z": (0.0, 0.05),
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.08, 0.08),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.05, 0.05),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = -50.0
        self.rewards.base_height_l2 = None
        self.rewards.track_base_height_command = RewTerm(
            func=mdp.track_base_height_command_exp,
            weight=3.0,
            params={
                "command_name": "base_height",
                "std": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.low_bar_clearance = RewTerm(
            func=mdp.low_bar_clearance,
            weight=-8.0,
            params={
                "bar_height": self.low_bar_height,
                "approach_distance": 0.45,
                "margin": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
                "bar_x": self.low_bar_x,
            },
        )
        self.rewards.low_bar_contact_penalty = RewTerm(
            func=mdp.low_bar_contact_penalty,
            weight=-80.0,
            params={
                "sensor_cfg": SceneEntityCfg("low_bar_contact"),
                "threshold": 0.05,
                "max_force": 20.0,
            },
        )
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -0.4
        self.rewards.lin_vel_xy_delta_l2.weight = -1.0
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_acc_l2.weight = -2.5e-6
        self.rewards.joint_pos_limits.weight = 0.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -0.5
        self.rewards.joint_pos_penalty.weight = -0.3
        self.rewards.joint_mirror.weight = -0.03
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height_body.weight = -3.0
        self.rewards.feet_height_body.params["target_height"] = -0.18
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.2

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.low_bar_contact = DoneTerm(
            func=mdp.low_bar_contact,
            params={
                "sensor_cfg": SceneEntityCfg("low_bar_contact"),
                "threshold": 25.0,
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        self.disable_zero_weight_rewards()


@configclass
class RCLowBarPlayEnvCfg(RCLowBarEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_height.height_range = (0.25, 0.25)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.rewards.is_terminated.weight = -200.0
        self.terminations.low_bar_contact.params["threshold"] = 0.2
