"""Shared 27dog adaptation for the RC locomotion tasks."""

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets.rc_wheeled import (
    RC27_INITIAL_HEIGHT,
    RC27_STANDING_HEIGHT,
    RC_WHEELED_CFG,
    WHEEL_MAX_VELOCITY,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg

LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
WHEEL_JOINT_NAMES = ["FR_feet_joint", "FL_feet_joint", "RR_feet_joint", "RL_feet_joint"]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES


def leg_action_sync(env, asset_cfg, joint_groups):
    """Compare leg action amplitudes using the explicit action order."""
    actions = env.action_manager.action[:, : len(LEG_JOINT_NAMES)]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_group in joint_groups:
        if len(joint_group) < 2:
            continue
        indices = [LEG_JOINT_NAMES.index(joint_name) for joint_name in joint_group]
        group_actions = torch.abs(actions[:, indices])
        reward += torch.mean(torch.square(group_actions - group_actions.mean(dim=1, keepdim=True)), dim=1)
    if joint_groups:
        reward /= len(joint_groups)
    asset = env.scene[asset_cfg.name]
    return reward * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


@configclass
class RCWheeledActionsCfg(ActionsCfg):
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=False,
        clip={".*": (-WHEEL_MAX_VELOCITY, WHEEL_MAX_VELOCITY)},
        preserve_order=True,
    )


def configure_wheeled_rc(env_cfg, initial_height=RC27_INITIAL_HEIGHT):
    initial_position = env_cfg.scene.robot.init_state.pos
    initial_joint_positions = env_cfg.scene.robot.init_state.joint_pos.copy()
    robot = RC_WHEELED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (initial_position[0], initial_position[1], initial_height)
    robot.init_state.joint_pos = initial_joint_positions
    robot.init_state.joint_pos[".*_feet_joint"] = 0.0
    robot.spawn.rigid_props.max_depenetration_velocity = (
        env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity
    )
    env_cfg.scene.robot = robot
    env_cfg.actions.joint_pos.joint_names = LEG_JOINT_NAMES.copy()
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.actions.joint_vel.joint_names = WHEEL_JOINT_NAMES.copy()

    for group in vars(env_cfg.observations).values():
        if group is None or not hasattr(group, "joint_pos"):
            continue
        for term_name, joint_names in (
            ("joint_pos", LEG_JOINT_NAMES),
            ("joint_vel", ALL_JOINT_NAMES),
            ("joint_effort", ALL_JOINT_NAMES),
        ):
            term = getattr(group, term_name, None)
            if term is not None:
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joint_names.copy(), preserve_order=True)

    for term_name in (
        "joint_pos_limits",
        "joint_pos_penalty",
        "stand_still",
        "joint_deviation_hip_l1",
        "joint_deviation_thigh_l1",
        "joint_deviation_calf_l1",
    ):
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES.copy(), preserve_order=True)

    for term_name in (
        "feet_air_time",
        "feet_slide",
        "feet_height",
        "feet_height_body",
        "feet_gait",
        "swing_feet_clearance",
        "wheel_vel_penalty",
    ):
        if hasattr(env_cfg.rewards, term_name):
            setattr(env_cfg.rewards, term_name, None)

    if env_cfg.rewards.base_height_l2 is not None:
        env_cfg.rewards.base_height_l2.params["target_height"] = RC27_STANDING_HEIGHT

    if env_cfg.rewards.action_sync is not None:
        env_cfg.rewards.action_sync.func = leg_action_sync

    reset_joints = env_cfg.events.randomize_reset_joints
    reset_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES.copy(), preserve_order=True)
    reset_joints.params["position_range"] = (-0.05, 0.05)
    env_cfg.events.reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES.copy(), preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    pose_range = env_cfg.events.randomize_reset_base.params["pose_range"]
    for angle in ("roll", "pitch"):
        lower, upper = pose_range.get(angle, (0.0, 0.0))
        pose_range[angle] = (max(lower, -0.05), min(upper, 0.05))
    env_cfg.disable_zero_weight_rewards()
