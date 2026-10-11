# 27dog 轮足 RC 训练说明

本次在 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/rc` 下创建独立轮足环境。轮足配置只共享通用 velocity 基类、MDP 函数和轮足适配工具，不再从四足 RC 配置导入任务文件，因此轮足与四足可以分别调整。

## 1. 直接启动 rough_stones

在已经安装本仓库和 Isaac Lab/Isaac Sim 的训练 Python 环境中，从仓库根目录运行。建议先用少量环境完成一次真实导入与训练检查：

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-RoughStones-RCwheel-v0 --num_envs 32 --max_iterations 1 --headless
```

正式训练：

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-RoughStones-RCwheel-v0 --num_envs 4096 --headless
```

如显存不足，降低 `--num_envs`。默认 15000 次迭代，日志和模型保存到 `logs/rsl_rl/rcwheel_rough_stones`，默认每 100 次迭代保存。单次迭代用于检查环境，不保证生成可用于回放的周期 checkpoint。

训练完成后回放：

```powershell
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-RoughStones-Play-RCwheel-v0 --num_envs 16
```

新策略的动作及观测维度改变，不可直接恢复旧四足的 12 维模型。LowWall/BrokenBridge 的加载实验名为 `rcwheel_rough_stones`；需要迁移时选取轮足 checkpoint，并显式启用 resume。

## 2. 关节与电机参数

训练资产为 `assets/rc_wheeled.py` 中的 `RC_WHEELED_CFG`，URDF 为 `data/Robots/RC/27dog urdf/urdf/27dog.SLDASM.urdf`。

| 项目 | 本次设置 |
| --- | --- |
| 12 个腿 joint 名称、origin、axis | 保留原有定义 |
| 腿 URDF hip 位置范围 | -0.9773843811168246 ~ +0.9773843811168246 rad |
| 腿 URDF thigh 位置范围 | -1.0471975511965976 ~ +4.1887902047863905 rad |
| 腿 URDF calf 位置范围 | -2.8012534494508987 ~ -0.7417649320975901 rad |
| 腿 URDF effort / velocity | 恢复旧版 53.5 N·m / 11 rad/s |
| 腿训练执行器 | 完整复制原 RC：hip 为 20 N·m / 30 rad/s；thigh、calf 为 30 N·m / 10 rad/s；PD、延迟、摩擦和 armature 均保留 |
| 四轮 joint | `FR/FL/RR/RL_feet_joint`，continuous，不设置 lower/upper，轴 +Y |
| 四轮 URDF/模拟器力矩上限 | 3.69 N·m |
| 四轮 URDF/模拟器速度上限 | 587 rpm × 2π/60 = 61.47049625524 rad/s |
| 额定力矩/速度，供后续保守设置参考 | 2.46 N·m；571 rpm = 59.794980172 rad/s |
| 轮速执行器 | ImplicitActuatorCfg，stiffness=0，damping=0.5，friction=0，armature=0.001 |
| 轮速动作 | scale=5 rad/s，最终速度目标裁剪至 ±61.47049625524 rad/s |

3.69 N·m 是图中的堵转/最大力矩，587 rpm 是空载速度。当前训练使用这两个值分别作峰值上限，未拟合速度—力矩曲线、热限制或 C620 电流模型，也不假设峰值力矩可以在空载极限转速下持续输出。PD damping 与 armature 是本次仿真起始设置，后续可通过实机/仿真响应校准。

图中参数按当前轮输出轴直接采用，未再乘一个未知减速比。若实机额外增加减速器，需要同时更新输出端力矩和速度。前进方向为四轮同号正速度；实机电机/编码器方向差异应在硬件映射中处理。

## 3. 动作、观测、初始化与奖励

动作共 16 维：前 12 维为腿位置偏移，后 4 维为轮速度。腿依次为 FR、FL、RR、RL，每腿 hip/thigh/calf；轮顺序也是 FR、FL、RR、RL。所有关节状态选择均显式保持顺序。

policy 和 critic 的 joint_pos 只包含 12 个腿关节，joint_vel 包含全部 16 个关节；critic 的 joint_effort 包含全部 16 个关节。原有动作历史随动作管理器自动扩为 16 维。这样策略能读取四轮速度，同时避免 continuous 轮角累积进入位置观测和站立/默认姿态惩罚。

27dog 的 16 个运动 link 恢复为 `*_hip_link / *_thigh_link / *_calf_link / *_feet_link`，joint 名保持原样，因此原有足端接触、材料随机化和 stance 奖励选择器仍能命中四个轮 body。17 个 mesh URI 改为相对 URDF 所在目录的 `../meshes/...`，不依赖旧 `27dog.SLDASM` ROS 包名解析；未重建 ROS/Gazebo 包元数据。本次保留 27dog 原有质量、惯量、碰撞体和轮轴位置。

rough_stones 默认腿角仍为 hip=0、thigh=0.7、calf=-1.5。默认基座 z 改为 0.48 m，平地名义轮胎最低点间隙约 57.65 mm，给重置扰动留出余量；站立高度奖励目标为 0.42 m。腿重置角扰动收紧到 ±0.05 rad，root roll/pitch 收紧到不超过 ±0.05 rad，其余任务已有的重置范围保留。轮子每次重置角度、速度为零，并与腿重置事件分开。

静态检查枚举了每腿三个关节以及 root roll/pitch 的扰动端点，rough_stones 全部基本碰撞体的最小平地间隙约 19.575 mm。这不是粗糙地形任意点上无穿透的证明，实际地形接触仍应由真实仿真检查。

低杆任务保留原前腿 thigh=1.1、后腿 thigh=1.4、calf=-2.45 的蹲伏角度，初始高度改为 0.28 m，base_height 命令改为 0.18~0.20 m。相同角扰动/姿态端点下，全部碰撞体最小平地间隙约 15.237 mm。低墙/断桥仍保留原障碍布局与进度奖励，起始 x/y 保留、z 更新为 0.48 m。

轮足版本取消 feet_air_time、feet_slide、feet_height、feet_height_body、feet_gait、swing_feet_clearance 和原 wheel_vel_penalty。原四足足端静止/腾空约束不适用于连续滚动，未将轮心世界速度当成滑移速度。腿的位置限位、默认姿态、stand_still 惩罚只选择 12 个腿；关节力矩、功率、加速度、速度限位等物理约束保留。rough_stones 的 stance_hip_zero、stance_calf_singularity、速度跟踪和非期望接触项保留。

低杆任务保留同类腿关节动作幅度同步奖励，但改用显式动作顺序索引前 12 维腿动作；原实现使用 articulation 关节编号索引动作，在腿位置与轮速度分组后可能读取错误维度。轮速不参与这项同步约束。原有 joint_mirror 的表达式已经限定到 hip/thigh/calf，继续保留。

## 4. 可用任务

| 任务 | 训练 ID | 回放 ID |
| --- | --- | --- |
| Flat | RobotLab-Isaac-Velocity-Flat-RCwheel-v0 | RobotLab-Isaac-Velocity-Flat-Play-RCwheel-v0 |
| Rough | RobotLab-Isaac-Velocity-Rough-RCwheel-v0 | RobotLab-Isaac-Velocity-Rough-Play-RCwheel-v0 |
| RoughStones | RobotLab-Isaac-Velocity-RoughStones-RCwheel-v0 | RobotLab-Isaac-Velocity-RoughStones-Play-RCwheel-v0 |
| LowBar | RobotLab-Isaac-Velocity-LowBar-RCwheel-v0 | RobotLab-Isaac-Velocity-LowBar-Play-RCwheel-v0 |
| LowWall | RobotLab-Isaac-Velocity-LowWall-RCwheel-v0 | RobotLab-Isaac-Velocity-LowWall-Play-RCwheel-v0 |
| BrokenBridge | RobotLab-Isaac-Velocity-BrokenBridge-RCwheel-v0 | RobotLab-Isaac-Velocity-BrokenBridge-Play-RCwheel-v0 |

这六类是 velocity 分支下原 RC 的全部任务，rough_stones 属于这个分支。本次没有迁移 vision 分支下的 Teacher/TS/DAgger 流程，也没有修改 sim2sim 的独立 MJCF。

## 5. 验证范围

已经执行：XML 解析、树连接和单位轴检查、12 个腿限位与旧 URDF 逐项一致检查、4 个 continuous 轮参数检查、17 个网格路径存在检查、新 Python 文件语法和格式检查、12 个训练/回放配置的离线实例化及 Gym 注册入口检查、动作/观测关节选择器检查、奖励 body/joint 正则匹配检查、原四足资产未被配置复制污染检查、初始化碰撞间隙计算，以及腿动作同步的离线数值检查（等幅、异幅和轮速隔离）。

离线实例化使用 Isaac Lab v2.3.2 官方 configclass；模拟器、资产和管理器依赖由替身对象提供，因此只验证配置继承和字段组合，不能视为 Isaac Sim 导入或 GPU 训练成功。本机没有安装 Isaac Lab/Isaac Sim，尚未执行真实 USD 导入、环境 step 或 PPO 训练。轮足资产设置 `force_usd_conversion=True`，真实训练启动时会重新转换当前 URDF，避免旧缓存影响；首次转换高面数 STL 可能耗时。

此前 `27dog_urdf_comparison_zh_CN.md` 是修复前审查记录，其中“全零限位/原命名/原 URI”描述的是修复前状态；本文件记录本次训练接入后的设置。
