# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import math
import sys
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.stair_slope import STAIR_SLOPE_CFG  # isort: skip


##
# Scene definition
##

#####################################################
#将simulation、scene、robot分开来定义，
#####################################################


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
########################################################
#与常规的使用：scene: InteractiveSceneCfg不同
#我们自己定义一个Scene类来配置我们自己所需要的场景
########################################################

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",#场景图的位置的表述，
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        #*************************************************
        #terrain type选择了生成器，其中选择的ROUGH_TERRAINS_CFG
        #在source\isaaclab\isaaclab\terrains\config\rough.py中定义，
        #会根据环境复杂度逐级提升障碍的难度，其中就有包含台阶等地形
        #************************************************

        #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # friction_combine_mode="multiply",
            # restitution_combine_mode="multiply",
            #静动摩擦力及弹性恢复系数

            #若在urdf中未在collision中设定<mu1>，则会使用默认的0.5
            # static_friction=1.0,
            # dynamic_friction=1.0,
            # restitution=1.0,
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.2,#决定物体碰撞后的弹性恢复能力，越低越硬
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
           #mdl_path则是相应的文件系统中的文件路径
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    #作为父类未专门设定

    # sensors
    #***************************************************************
    #height_scanner用于观测前方地形的高度变换
    #若是训练中配置了这个观测器，则需要在实机中添加深度相机或是雷达进行预测
    #在实机中需要考虑
    #***************************************************************


    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    #***************************************************************
    #height_scanner_base用于观测机器人底部的高度
    #例如：返回 0.0（平坦地面）或 -0.2（凹陷）或 0.3（凸起）
    #在后续奖励中是用于鼓励机器人站在平坦地面上
    #*****这个是鼓励对抗地形，即在不平的地面保持身形是水平的，后续可利用imu对其改进，使其适应地形***************************
    #***************************************************************
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    #***************************************************************
    #在奖励函数中留有了flat_orientation_l2的保持机体水平的接口，若要启用，需要配置imu传感器的设置
    #https://docs.robotsfan.com/isaaclab/source/overview/core-concepts/sensors/imu.html
    #确保在urdf中imu_link的设定，若存在也额外需要配置prim_path
    #Isaac Lab的IMUCfg配置会直接影响训练，而URDF中的配置只影响ROS话题发布。
    #***************************************************************


    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    #需要修改训练地形设置
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 15.0),#在指令改变前的时间范围，采样范围
        #设计环境配置
        rel_standing_envs=0.04,#控制机器人站立状态的环境比例，0.02 表示 2% 的环境，在4096个并行环境中，大约有82个环境会处于站立状态
        rel_heading_envs=1.0,#所有环境都会接受朝向命令
        heading_command=True,#都是追踪线性xy，True：目标朝向角度指令；False：目标角速度指令
        heading_control_stiffness=0.5,#转向的相应强度
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),#速度指令的分布范围
        #初始的速度配置
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    #*************************************************************************
    #将关节位置设定为action输出，其中的若是见到与action_manager相关的配置说的都是关节位
    #指的是一个范围[-scale, scale]，而非绝对位置
    #*************************************************************************

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    )
    #读取urdf并使用其中的关节进行运动控制
    #use_default_offset：是否使用原本urdf中定义的初始位置作为偏移量
    #preserve_order=True 保持URDF中定义的关节顺序
    #scale用作动作缩放# 策略输出范围：[-1, 1]
                     # scale = 0.5 时，实际关节变化范围：[-0.5, 0.5] 弧度
                    #************************************************************* 
                    #经过scale后得到的是单步输出的范围，也就是若是100hz，则最高可达50
                    #************************************************************
    #clip则是对动作进行裁剪，避免超出范围，将超出部分改为clip的上限
    #处理：策略输出→ clip → scale
@configclass
class ObservationsCfg:
    #状态空间的配置
    """Observation specifications for the MDP."""
    
    # 观测空间设计原则：
    # observation_design_principles = {
    #     "必要性": "只包含策略决策必需的信息",
    #     "维度控制": "避免观测空间过大导致训练困难", 
    #     "信息冗余": "避免重复信息",
    #     "实机可行性": "确保实机能提供相同观测"
    #故不是所有sensor都要加入观测空间中，可以加入奖励函数间接引导
    #在观测空间中没有使用的sensor，在实机中并不是必须配置

    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    #其中，contact不作为观测输入，只是训练过程的参考量，
    #可考虑加入带噪声的imu sensor和不带噪声，以0/1为输出的足底传感器，并将其加入观测变量
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        #每个函数的配置查看Observations的规定
        #https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.observations

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
            #history_length=5,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        #评论家网络
        #观测对象是一样的，但是没有设定噪声

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )

        base_pos_z = ObsTerm(
            func=mdp.base_pos_z,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )



        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=5,
            flatten_history_dim=True,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
            #history_length=5,
            flatten_history_dim=True,
        )



        #可观测力，或是对力进行状态价值评估
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            clip=(-100, 100),
            scale=0.01,
            history_length=5,
            flatten_history_dim=True,

        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    #policy作为Actor网络的输入
    
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    #https://docs.robotsfan.com/isaaclab/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.events
    #
    #  startup

    #物理属性随机变化
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            #将连续范围离散为64个等级",
            #"内存优化": "减少随机化带来的内存开销",
            #"训练稳定性": "避免过于细粒度的变化",
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        #只对躯干进行随机质量，单位是kg
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        #对其他部位，如四肢

        #整腿重约2.778kg
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",#以缩放的方式
            "recompute_inertia": True,
        },
    )

    # Skip: inertia updated via mass randomization by setting recompute_inertia=True
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    randomize_com_positions = EventTerm(
        #质心位置偏移
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.08), "y": (-0.05, 0.05), "z": (-0.02, 0.04)},
        },
    )


    # 关节物理参数随机化（摩擦 + armature）
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.2, 1.5),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        #只在reset初始阶段对躯干施加外力和扭矩，
        #*****可用于优化专门的起步稳定性**********
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(
        #func=mdp.reset_joints_by_scale,
        #提供以加法形式及的
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.5, 0.5),#是否太大？
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.6, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )



    randomize_reset_base = EventTerm(
        #基座初始状态的随机化
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-3.14, 3.14)},

            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval

    #与 randomize_apply_external_force_torque 相辅相成
    #用作在运动过程中随机施加推力，力的体现形式是以速度的形式出现
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",#指运动过程中
        interval_range_s=(8.0, 12.0),#时间间隔
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5) ,"roll": (-0.3, 0.3), "pitch": (-0.3, 0.3),}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    #奖惩权重确定的，正的是鼓励，负的是惩罚

    # GeneralPenalize z-axis base linear velocity using L2 squared kernel.
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    #对非达到限制时间就提前终结（失败的）进行惩罚
    
    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    #z方向约束，减少z方向的速度，鼓励机器人保持在地面上

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    #惩罚roll和pitch的角速度过大
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    #惩罚非水平面（现在姿态非平衡的）
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    #****************************************************
    #不可参考base_height_l2的写法，在该函数原先的定义并没有接受sensor_cfg的配置
    #加入对imu的观测作为输入进行水平姿态的奖励，需要自己写
    #****************************************************
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

    lin_vel_xy_delta_l2 = RewTerm(
        func=mdp.lin_vel_xy_delta_l2, 
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot")
        },)


    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    ##未防止突然间的过大加速度变化，只是针对不给大加速度在引入imu后可启用
    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )#关节加速度惩罚，只是限制大小，没有限制突变

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        #attr_name作为变量名，不用单独设定奖励函数
        setattr(self, attr_name, rew_term)
        #首先这是一个针对关节位置偏差的奖励函数，需要注意使用场景，常见于站立类任务中
        #其次这个函数的设计是可以按照关节类型来分别设定权重强度，其中是使用索引确认关节类型。
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    #limit类的惩罚函数会自动读取URDF中的限制

#################################以上均为issac_lab自带的奖励函数#####################################

    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    #通过功率约束来优化机器人能耗的核心机制。需要注意数量级

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.0,
        params={
            "command_name": "base_velocity",#要监控的指令名称
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    #当指令非常小的时候，若是关节发生了移动，则进行惩罚；总结就是不接受过小的速度指令

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.2,
            "command_threshold": 0.1,
        },
    )
    #总体而言是重力依赖机制是鼓励机器人在运动过程中保持身体水平

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )
    #检索镜像组的关节角度是否是保持一致的
    #*************************************************
    #若是出现同一组的两腿差别特别大，可适当加强
    #*************************************************


    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )
    #检索镜像组的动作输入是否是保持一致的，适合平地行走
    #未启用

    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )
    #统一同类关节的运动幅度，避免单独一只腿突然幅度特别大，更适合平面行走的任务

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    #惩罚实际施加力矩与预期计算力矩之间的偏差，而不是直接限制力矩上限。
    #此项目的控制action项仍是使用action的，所以我们不能使用这个以力作为反馈的奖励

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)#前后两个action的变化率
    #鼓励其不要突变
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    #根据预设的接触时间，若比threshold要大则是意外接触

    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},
    )
    #对施加超过threshold大小的力足底力金星和惩罚，单位是N

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    #这两项奖励，由于使用的是高斯核函数，所以所需要给的权重是正的，本质上是对未追踪的惩罚

    #误差越小，奖励越大。

    # Others
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )
    #对腿滞空的鼓励

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )
    #惩罚四足机器人各脚部在空中和地面时间的不一致性，促进步态对称性和协调性

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    #这个 GaitReward 奖励项正是用于调整四足机器人的步态，使其能够实现斜对角两腿同步运动（即小跑步态）

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )
    #惩罚脚部接触数量与期望值不符的情况，确保机器人在运动时保持特定的支撑模式

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )
    #计算触地脚数量，鼓励多脚接触地面，不建议启动

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )
    #确认落足点方向的正确性，避免机器人出现绊倒现象
    #*********************************************
    #在修改足底坐标系方向前不可使用
    #*********************************************


    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    #检验侧滑，需要结合rough_env_cfg理解

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,

            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            #相对于基坐标的高度

            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )


    # feet_distance_xy_exp = RewTerm(
    #     func=mdp.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )
    #对落足点进行鼓励或惩罚，可动态设计落足点的长度

    upward = RewTerm(func=mdp.upward, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #超时

    #若是启用终止

    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    #检测是否超出地形边界，超出则终止；设置的distance_buffer是距离缓冲值

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    #限制足底触地力的大小。通过threhold（单位N）限制
    #非触地点

    dead = DoneTerm(
        func=mdp.dead_down,
        params={"threshold": 0, "asset_cfg": SceneEntityCfg("robot")},
    )
    #当机器人完全倒下（四脚朝天）时终止：threshold=0.9 对应约154°倾斜
    #可调整 threshold: 0.8(约143°) | 0.9(约154°) | 0.95(约162°)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    #索引source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\terminations.py
    #其中是根据当前机器人所行走的距离来间接评价当前的效果
    #随后通过move_up和move_down来进行调整
    #缩影source\isaaclab\isaaclab\terrains\terrain_importer.py中的update_env_origins函数
    #设定地形难度，难度的对应是通过source\isaaclab\isaaclab\terrains\config\rough.py
    #对于每种类型的障碍，都是同样的占比，难度主要影响其高度，其中每个障碍都配置了一个范围，系统会自动根据level进行差值

    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
        },
        #根据track_lin_vel_xy_exp对速度指令的追踪效果来提升或降低指令范围
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        #下采样率策略每4个物理步执行一次 | 影响 render_interval 和传感器更新周期

        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005#仿真步长
        self.sim.render_interval = self.decimation#渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)

        #此为对scene的定义，用于训练过程值中，其中contact_force是作为训练中间项，在最后的训练出来的模型中不作为传入参数，但在训练中需要用到，所以也得设定
        #此处的none是对应是否定义了传感器
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if reward_attr is not None and not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


def create_obsgroup_class(class_name, terms, enable_corruption=False, concatenate_terms=True):
    #未查明在何处使用
    
    
    #用于动态创建观测组配置类的工厂函数
    
    """
    Dynamically create and register a ObsGroup class based on the given configuration terms.

    :param class_name: Name of the configuration class.
    :param terms: Configuration terms, a dictionary where keys are term names and values are term content.
    :param enable_corruption: Whether to enable corruption for the observation group. Defaults to False.
    :param concatenate_terms: Whether to concatenate the observation terms in the group. Defaults to True.
    :return: The dynamically created class.
    """
    # Dynamically determine the module name
    module_name = inspect.getmodule(inspect.currentframe()).__name__

    # Define the post-init function
    #后初始化函数
    def post_init_wrapper(self):
        setattr(self, "enable_corruption", enable_corruption)
        setattr(self, "concatenate_terms", concatenate_terms)

    # Dynamically create the class using ObsGroup as the base class
    terms["__post_init__"] = post_init_wrapper
    dynamic_class = configclass(type(class_name, (ObsGroup,), terms))

    # Custom serialization and deserialization
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Add custom serialization methods to the class
    dynamic_class.__getstate__ = __getstate__
    dynamic_class.__setstate__ = __setstate__

    # Place the class in the global namespace for accessibility
    globals()[class_name] = dynamic_class

    # Register the dynamic class in the module's dictionary
    if module_name in sys.modules:
        sys.modules[module_name].__dict__[class_name] = dynamic_class
    else:
        raise ImportError(f"Module {module_name} not found.")

    # Return the class for external instantiation
    return dynamic_class
