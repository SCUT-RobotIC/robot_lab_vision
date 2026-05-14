# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
纯 DAgger 蒸馏环境配置（无奖励函数版本）
学生通过监督学习模仿教师，不使用 RL 奖励
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import robot_lab.tasks.manager_based.locomotion.vision.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


##
# 观测项函数
##

def camera_rgb_image(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """学生的视觉观测"""
    camera = env.scene.sensors[sensor_cfg.name]
    rgb = camera.data.output["rgb"]
    
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    
    return rgb.permute(0, 3, 1, 2)  # (N, 3, H, W)


def teacher_privileged_state(env) -> torch.Tensor:
    """
    教师的完整特权观测
    包含所有教师 PolicyCfg 中的观测项
    """
    robot = env.scene["robot"]
    obs_list = []
    
    # 基础运动信息
    obs_list.append(robot.data.root_lin_vel_b)
    obs_list.append(robot.data.root_ang_vel_b)
    obs_list.append(robot.data.root_pos_w[:, 2:3])
    obs_list.append(robot.data.projected_gravity_b)
    
    # 速度指令
    velocity_command = env.command_manager.get_command("base_velocity")
    obs_list.append(velocity_command)
    
    # 关节状态
    obs_list.append(robot.data.joint_pos)
    obs_list.append(robot.data.joint_vel)
    obs_list.append(robot.data.applied_torque)
    
    # 最后的动作
    obs_list.append(env.action_manager.action)
    
    # 高度扫描
    height_scanner = env.scene.sensors["height_scanner"]
    obs_list.append(height_scanner.data.ray_hits_w[..., -1].unsqueeze(-1))
    
    # 脚底摩擦系数（简化）
    contact_sensor = env.scene.sensors["contact_forces"]
    feet_friction = contact_sensor.data.net_forces_w
    obs_list.append(feet_friction)
    
    # 地形信息
    terrain_level = env.terrain.terrain_levels.float() / env.terrain.terrain_generator.curriculum_length
    obs_list.append(terrain_level.unsqueeze(-1))
    
    terrain_type = env.terrain.terrain_types.float()
    obs_list.append(terrain_type.unsqueeze(-1))
    
    return torch.cat(obs_list, dim=-1)


##
# Scene configuration
##

@configclass
class StudentSceneCfg(InteractiveSceneCfg):
    """学生场景配置（添加相机）"""
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.2,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    robot: ArticulationCfg = MISSING
    
    # ========== 视觉传感器 ==========
    camera_wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/wrist_camera",
        update_period=0.1,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.08),
            rot=(0.7071, 0.0, -0.7071, 0.0),
            convention="ros",
        ),
    )
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# 观测配置
##

@configclass
class ObservationsCfg:
    """观测配置（学生 + 教师）"""

    @configclass
    class StudentPolicyCfg(ObsGroup):
        """学生观测：RGB + 本体感觉"""
        
        camera_rgb = ObsTerm(
            func=camera_rgb_image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist")},
        )
        
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=3,
            flatten_history_dim=True,
        )
        
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=3,
            flatten_history_dim=True,
        )
        
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=3,
            flatten_history_dim=True,
        )
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=3,
            flatten_history_dim=True,
        )
        
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=3,
            flatten_history_dim=True,
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False  # Vision 不拼接


    @configclass
    class TeacherCfg(ObsGroup):
        """教师观测：完整特权状态"""
        
        privileged_state = ObsTerm(
            func=teacher_privileged_state,
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: StudentPolicyCfg = StudentPolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


##
# 动作、指令、终止配置（与教师一致）
##

@configclass
class CommandsCfg:
    """命令配置"""
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 15.0),
        rel_standing_envs=0.04,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(-1.0, 1.0), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """动作配置"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True, 
        clip=None, 
        preserve_order=True
    )


@configclass
class TerminationsCfg:
    """终止条件"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    
    dead = DoneTerm(
        func=mdp.dead_down,
        params={"threshold": 0, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# 环境配置
##

@configclass
class LocomotionVisionDAggerEnvCfg(ManagerBasedRLEnvCfg):
    """
    纯 DAgger 蒸馏环境配置
    ❌ 不使用 RL 奖励
    ✅ 通过监督学习模仿教师
    """

    scene: StudentSceneCfg = StudentSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # ❌ 删除 rewards 和 curriculum
    # rewards: RewardsCfg = None
    # curriculum: CurriculumCfg = None
    
    @configclass
    class DistillationCfg:
        """DAgger 配置"""
        
        # 教师策略路径
        teacher_policy_path: str = "logs/rsl_rl/teacher_experiment/exported/policy.pt"
        
        # DAgger 参数
        mode: str = "dagger"  # 纯 DAgger 模式
        beta_schedule: str = "linear"  # 查询概率衰减方式
        beta_init: float = 1.0  # 初始 100% 查询教师
        beta_final: float = 0.1  # 最终 10% 查询教师
        beta_decay_steps: int = 1000  # 衰减步数
        
        # 损失函数配置
        loss_type: str = "mse"  # 行为克隆损失（MSE）
        use_task_reward: bool = False  # ❌ 不使用任务奖励
        
        # 数据收集
        collect_interval: int = 1  # 每步都收集
        buffer_size: int = 1000000
        save_buffer_path: str = "datasets/dagger_buffer.zarr"
        
    distillation: DistillationCfg = DistillationCfg()

    def __post_init__(self):
        """后初始化"""
        self.decimation = 4
        self.episode_length_s = 20.0
        
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        if self.scene.camera_wrist is not None:
            self.scene.camera_wrist.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt