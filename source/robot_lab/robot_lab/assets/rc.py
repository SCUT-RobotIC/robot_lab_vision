# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
#???????????????????????????????????????????????????????????????????
#sim_utils作为一个统一的数据格式收集接口，可研究源码了解其还提供了什么
#???????????????????????????????????????????????????????????????????
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


RC_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,#是否浮动基
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,#圆柱碰撞体积是否更改为胶囊体积
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/a1_description/urdf/a1.urdf", #暂时不换
        
        #通过在data中配置urdf，在assets中配置CFG文件，不通过launch修改
        
        activate_contact_sensors=True,
        #************************************************************************************
        #此处作为传感器的启用，需要注意：在issac——lab中，对于传感器的定义是在urdf中的<contact>中定义的，
        #Isaac Lab的IMUCfg配置会直接影响训练，而URDF中的配置只影响ROS话题发布。
        #urdf中的gazebo定义是用于辅助ros话题的发布，更多体现在实机部署上的影响
        #************************************************************************************
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #用于设定刚体的物理属性
            disable_gravity=False,
            retain_accelerations=False,#不保留上一帧的加速度，每帧重新计算物理，确保准确性
            linear_damping=0.0,
            angular_damping=0.0,
#           功能：针对整个刚体的线性和角速度阻尼系数，而不是单个电机
#           交互：设为0表示无空气阻力，运动更自然
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        #配置关节属性，分别代表禁用关节部件之间的自碰撞检测，提高仿真性能；**位置求解器迭代次数，影响关节约束的精度**；速度求解器迭代次数，设置为0表示使用默认值

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        #配置关节去驱动器的PD增益，在urdf转换过程中都设定为0作为默认值，用于做分割，后续的控制中会有新的控制器顶替，在下方actuators有额外设定
    ),
    
    #启动姿态，并非0位
    #****************************************************************
    #而urdf的0位置的定义必须是如同于宪元的论文中所描述的四肢垂直向下的姿态
    #此处与力控狗有所不同，在处理力控狗时，我们时将爬姿作为电机机械0位，再通过函数将其和原理所需的0位进行偏移
    #****************************************************************
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),#此处为为何这是从高处落下的。
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.7,
            "R.*_thigh_joint": 0.7,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    #软关节限制因子，防止关节超出物理限制
    # actuators={
    #     "legs": DCMotorCfg(#对直流电机模型的模拟
    #         joint_names_expr=[".*_joint"],
    #         effort_limit=18,#最大输出力矩限制
    #         saturation_effort=26,#力矩饱和限制（峰值）
    #         velocity_limit=17.0,
    #         stiffness=20.0,
    #         damping=0.5,
    #         friction=0.0,
    #     ),
    # },


    actuators={
        "hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=20,#最大输出力矩限制
            #saturation_effort=26,#力矩饱和限制（峰值）
            velocity_limit=30.0,
            stiffness=25,
            damping=2,
            friction=0.05,
            armature=0.02,
            min_delay=2,
            max_delay=7, 
        ),
        "thigh": DelayedPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=30,#最大输出力矩限制
            #saturation_effort=26,#力矩饱和限制（峰值）
            velocity_limit=10.0,
            stiffness=25,
            damping=2,
            friction=0.05,
            armature=0.02,
            min_delay=2,
            max_delay=7, 
        ),
        "calf": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=30,#最大输出力矩限制
            #saturation_effort=26,#力矩饱和限制（峰值）
            velocity_limit=10.0,
            stiffness=25,
            damping=2,
            friction=0.05,
            armature=0.02,
            min_delay=2,
            max_delay=7, 
        ),
    },
)
