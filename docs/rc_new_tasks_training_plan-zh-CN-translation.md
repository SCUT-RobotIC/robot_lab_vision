# RC 新移动任务训练计划

该笔记记录了三个新的 RC 机器人任务的建议训练计划：

1.  低杆下蹲的身体高度指令，
2.  跨越高度约为 0.30 米的低墙，
3.  在凹凸不平的石头状地面上行走。

机器人已经具有平地、斜坡和楼梯行走策略。推荐的起始点是速度 RC 堆栈：

*   `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/rc/rough_env_cfg.py`
*   `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/rc/agents/rsl_rl_ppo_cfg.py`
*   `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

对于真实机器人迁移，将可部署角色的观察限制在硬件上可用的信号：角速度、投影重力、速度指令、关节位置、关节速度和先前动作。将地形高度扫描、真实基线线性速度、接触、地形类型和扭矩等特权信息保留在评论家或教师中。

## 常见策略

尽可能使用现有的训练好的行走策略作为初始化。

推荐步骤：

1.  使用非对称 Actor-Critic 使用 PPO 进行训练或微调。
2.  Actor 只观察可部署的观测。
3.  评论家观察特权模拟信号。
4.  如果特权评论家不够，训练一个特权教师并蒸馏成一个可部署的 MLP 学生。
5.  仅导出演员/学生以进行部署。

在开始时不要将所有三个新任务以全难度混合。首先训练三个专门策略，然后可选地通过课程或多任务命令将它们合并。

## 任务 1：低杠下蹲的躯干高度指令

### 已完成初版
`
python scripts/reinforcement_learning/rsl_rl/train.py   --task RobotLab-Isaac-Velocity-LowBar-RC-v0   
`

目标：添加一个指令的躯干高度，使机器人能够降低其躯干并通过一个高度限制的横杆，同时仍然跟踪速度。

重要比赛约束：低杆不是固定住的。如果机器人碰到并把杆碰掉，这个障碍就失败。因此这个任务应该按“严格无接触通过”来训练，而不是按“允许擦碰固定障碍物”的限高任务来训练。

### 指令设计

扩展速度指令从 `[vx, vy, yaw]` 以包括目标躯干高度：

```text
[vx, vy, yaw, base_height_cmd]
```

使用保守的指令范围首先：

```text
base_height_cmd: 0.22 m to 0.35 m
normal standing: about 0.30 m to 0.35 m
ducking phase: about 0.22 m to 0.28 m
```

实现选项：

*   在 `velocity/mdp/commands.py` 中添加一个新的指令生成器，扩展 `UniformThresholdVelocityCommand` 。
*   或者添加一个名为 `base_height` 的第二个指令术语，并通过 `mdp.generated_commands` 连接到观察中。

如果现有的速度命令需要与旧政策保持兼容，那么第二个选项更干净。

### 环境设计

在大量随机化之前，从一个简单的低门槛场景开始。

训练阶段：

1.  没有物理栏杆，只有高度指令跟踪。
2.  添加一个带有足够间隙的静态参考栏杆，并在任意机器人-栏杆接触时终止。
3.  替换或额外验证为轻质动态栏杆，使其可以被碰动。
4.  随机化栏杆的高度、宽度、半径和位置。
5.  混合正常行走和蹲伏的片段。

几何形状不应成为演员观察的一部分，除非真实机器人具有匹配的感知信号。如果真实任务是命令驱动的，操作员或上游规划器应在障碍物之前命令下蹲高度。

实现说明：静态栏杆加“接触即终止”是一个很好的第一阶段训练近似，因为它能给出干净的失败信号。但在信任策略之前，应该用轻质动态栏杆验证，并在发生接触或栏杆位移超过小阈值时终止。这样可以避免策略在仿真中学会“轻轻蹭杆”。

### 奖励

添加或重用基础高度跟踪奖励：

```text
r_height = exp(-(base_z - base_height_cmd)^2 / sigma^2)
```

尽可能使用局部地形相对基础高度。现有的 `base_height_l2` 可以从固定目标改编为命令目标。

建议的奖励更改：

*   正面：身体高度跟踪 `base_height_cmd` 。
*   正面：速度跟踪在低高度指令期间保持激活但略微降低权重。
*   正面：如果训练中可以获得栏杆姿态作为特权信息，则奖励身体相对栏杆的安全通过余量。
*   负面：俯仰/滚转过大。
*   负面：膝盖/髋关节限制接近。
*   负面或终止：任意机器人部位接触栏杆。
*   终止：栏杆位移或旋转超过允许阈值。
*   负面：大幅度动作率和关节加速度。

避免在蹲起时对关节偏差过度处罚。低姿态需要较大的腿部配置变化，因此 `joint_pos_penalty` 和 `stand_still` 可能需要命令相关的权重。

不要只依赖较小的接触惩罚。由于比赛中碰杆即失败，栏杆接触应该直接终止 episode，或者至少给非常大的惩罚。建议模式是：

```text
成功奖励 = 通过栏杆所在平面 + 无栏杆接触 + 栏杆无明显位移
失败终止 = 任意栏杆接触，或栏杆位移 > 阈值
```

### 课程

建议课程：

1.  `base_height_cmd` 范围：0.30 米至 0.35 米。
2.  扩展至 0.26 米至 0.35 米。
3.  扩展至 0.22 米至 0.35 米。
4.  添加无接触低杠，高度为 0.38 米至 0.45 米。
5.  先要求较大的安全余量，例如身体低于栏杆 5 cm 至 8 cm。
6.  逐步减小间隙，直到与目标真实低杠任务匹配。
7.  使用轻质动态栏杆和严格位移终止进行验证。

成功指标：

*   在低杠下通过且零机器人-栏杆接触，
*   栏杆位移低于比赛安全阈值，
*   跟踪前进速度，
*   通过低段后恢复到正常高度，
*   没有膝盖/臀部饱和峰值。

### 真实机器人笔记

这是在没有深度相机的情况下三个任务中最可行的任务，因为障碍物可以通过命令调度来处理。如果操作员或规划者告诉它何时降低，策略就不需要检测横杆。

由于真实栏杆会被碰掉，实机部署时应使用保守的高度指令。机器人应比名义测得的通过高度再低一些，为状态估计误差、身体振荡、执行器延迟和小幅地形高度变化留出余量。

## 任务 2：跨越约 0.30 米的低墙

目标：使机器人跨越许多约正常站立高度的低墙。

这是在没有地形感知的情况下最难的任务。0.30 米的墙不仅仅是粗糙行走；它是一个需要脚部间隙、身体俯仰管理和可靠接触转换的障碍物协商任务。

### 地形设计

创建新的墙壁地形生成器或自定义网格地形。

墙壁随机化：

```text
height: 0.10 m -> 0.30 m curriculum
thickness: 0.08 m -> 0.25 m
top shape: flat first, then rounded/chamfered
spacing: 1.0 m -> 3.0 m
orientation: mostly perpendicular first, then small yaw variation
```

从每个地形瓦片上开始只有一个墙壁。然后只有在策略可靠地清除一个墙壁之后，才增加到每个瓦片上有多个墙壁。

### 观测设计

对于模拟训练，使用特权评论家或教师观测：

*   地形高度扫描，
*   脚部接触状态，
*   基础高度，
*   如果可用，本地地形类型或墙壁距离。

对于没有深度摄像头的可部署角色/学生：

*   速度指令，
*   IMU 角速度，
*   投影重力，
*   关节位置和速度，
*   先前动作，
*   可选地，如果真实机器人有脚接触传感器。

如果没有感知和没有墙壁计时命令，最终的实时策略将不知道何时准备墙壁。在这种情况下，它只能学习一种慢速、反应式、高离地面的步态。这可能适用于分散的低障碍物，但对于真正的垂直墙壁则不太可靠。

推荐的命令添加：

```text
obstacle_mode or step_height_cmd
```

一个简单的离散命令就足够了：

```text
0 = normal walking
1 = obstacle crossing / high-step gait
```

这比期望一个没有感知能力的 MLP 在撞击前推断出墙壁更现实。

### 奖励

保持速度跟踪，但在靠近墙壁时降低其优先级。添加特定于障碍物的术语：

*   正面：穿过墙壁的前进。
*   正面：脚在挥杆过程中越过障碍物的高度清晰。
*   正面：基座在穿越过程中始终保持高于最低高度。
*   正面：所有脚最终都到达了远侧。
*   负面：身体与墙壁发生碰撞。
*   负：接触力过大。
*   负：脚在墙面拖拽。
*   负：滚转/俯仰超出安全阈值。
*   负：关节功率高且动作冲击大。

现有的有用奖励：

*   `feet_air_time`
*   `feet_height`
*   `feet_height_body`
*   `contact_forces`
*   `action_rate_l2`
*   `joint_power`
*   `flat_orientation_l2`

可能需要的新奖励：

*   `wall_clearance_feet`
*   `wall_progress`
*   `body_wall_collision`
*   `front_feet_far_side`
*   `rear_feet_far_side`

### 课程

建议的课程：

1.  从现有的楼梯/斜坡政策开始。
2.  在 0.05 米至 0.10 米处添加低块。
3.  增加到 0.15 米。
4.  增加到 0.20 米，并加宽顶部表面。
5.  增加到 0.25 米。
6.  仅当步态稳定后尝试 0.30 m。

也要调整指令速度：

```text
early: vx = 0.1 m/s to 0.3 m/s
later: vx = 0.2 m/s to 0.6 m/s
```

早期避免高横向速度。

### 师生选项

这项任务非常适合教师-学生训练：

```text
teacher: MLP with height_scan / wall distance / terrain type / contact
student: MLP with deployable proprioception and optional obstacle_mode command
```

如果真实机器人没有感知能力，请在学生中包含一个障碍物指令。否则学生将无法知道墙壁即将到来。

### 真实机器人注意事项

将 0.30 米视为激进。在真实试验前：

*   验证开环最大脚部间隙，
*   检查膝盖和髋部扭矩余量，
*   在模拟中添加紧急身体接触终止，
*   用 0.05 米至 0.10 米的障碍物开始实际测试，
*   使用安全龙门架或软泡沫障碍物。

## 任务 3：跨越不平整的石头状地面

目标：使机器人能够跨越类似于石子路的随机凸起/凹陷。

这是当前崎岖地形框架的最佳匹配。它不需要精确的障碍物时间，如果地形不是过于不连续，本体感觉策略可以学习稳健的反应。

### 地形设计

使用或扩展 `HfRandomUniformTerrainCfg` 。

初始地形范围：

```text
noise_range: 0.01 m to 0.05 m
noise_step: 0.01 m to 0.02 m
horizontal_scale: 0.05 m to 0.10 m
```

硬地形范围：

```text
noise_range: 0.05 m to 0.12 m
noise_step: 0.02 m to 0.04 m
```

对于石质地形，最好选择混合：

*   随机均匀的高度场，
*   小的离散步骤，
*   低的金字塔形凸起，
*   稀疏的凸起斑块，
*   摩擦变化。

避免起始处出现极端的尖锐高度不连续性；它们通常会产生不真实的接触冲量，并导致模拟到现实的转换效果差。

### 观察设计

角色可以保持可部署状态：

*   无高度扫描
*   除非从状态估计中可用，否则没有真正的基线线性速度，
*   保持 IMU、关节状态、命令、先前的动作。

评价者可以保持：

*   高度扫描，
*   基础线性速度，
*   基础位置 z，
*   接触力，
*   共同努力。

这正是 RC 速度堆中已经存在的非对称演员-评论家模式。

### 奖励

重用大部分当前的粗略行走奖励，并进行以下调整：

*   逐步增加鲁棒性项，而不是一次性全部增加。
*   保留 `track_lin_vel_xy_exp` 和 `track_ang_vel_z_exp` 。
*   保持适度的 `base_height_l2` ，但在崎岖地形上不要强迫一个完美的固定高度。
*   保持 `flat_orientation_l2` 、 `ang_vel_xy_l2` 和 `lin_vel_z_l2` 。
*   如果冲击剧烈，增加 `contact_forces` 的惩罚。
*   如果脚部打滑变得明显，保持 `feet_slide` 。
*   保留 `action_rate_l2` 和 `joint_acc_l2` 以实现更平滑的真实行为。

潜在的新奖励：

*   `foot_clearance_adaptive` : 奖励摆动脚部清除度相对于局部地形高度，仅使用特权地形数据作为奖励。

### 课程

建议的课程：

1.  混合 70%平面/斜坡/楼梯，30%轻微随机粗糙。
2.  将随机粗糙比例增加到 50%。
3.  增加凸起振幅。
4.  添加摩擦随机化。
5.  添加推送干扰。
6.  减少地形缓存确定性，以便策略不会记住模式。

对于实际部署，这应该作为现有粗糙策略的扩展进行训练，而不是从头开始。

### 真实机器人笔记

这项任务在没有深度相机的情况下最为真实。机器人可以通过 IMU 和关节/接触动力学进行反应。如果脚部接触传感器可用，将二进制脚部接触添加到演员观察中可能会显著提高在松软或不平整表面上的性能。

## 推荐优先级

1.  石质不平整地面。感知要求最低，最适合现有的粗糙移动。
    
2.  身体高度指令用于低姿态蹲伏。如果低姿态由操作员或规划者指令，则可行。
    
3.  低墙攀爬。风险最高。强烈建议使用感知或显式的障碍跨越指令。只有在教师拥有特权墙/高度信息且学生至少获得模式或时间指令时，才使用师生模式。
    

## 实施检查清单

共享：

*   保留演员观察部署。
*   保留评论家特权。
*   从现有的 RC 粗糙/平坦检查点进行微调。
*   一次添加一个任务。
*   在正式部署前导出并验证观察/行动顺序。

## 服务器继续训练工作流：从部署导出的 `policy.pt` 开始

这部分用于服务器端继续低墙任务开发，前提是你手上只有部署工程里导出的 `policy.pt`，而没有原始训练日志中的 `model_*.pt` checkpoint。

### 结论先说

*   `policy.pt` 不能直接当作 `rsl_rl` 的恢复训练 checkpoint 使用。
*   原因是它通常只包含部署所需的 actor，以及可能存在的 actor normalizer。
*   它通常不包含 critic、PPO optimizer 状态、训练迭代状态等继续训练所需信息。
*   因此不要把 `policy.pt` 复制到 `logs/` 后直接使用 `--resume` 指向它。
*   正确做法是：把 `policy.pt` 当作 actor 的预训练初始化，再启动一个新的 PPO 训练。

### 为什么 `policy.pt` 不能直接 `--resume`

当前工程的 `rsl_rl` 训练脚本恢复逻辑会去加载标准训练 checkpoint，也就是类似：

```text
model_100.pt
model_500.pt
model_1000.pt
```

这类文件来自训练过程中的保存，通常位于：

```text
logs/rsl_rl/<experiment_name>/<run_name>/model_*.pt
```

而部署导出的 `policy.pt` 是给推理使用的 TorchScript/JIT 模型。它更接近“只保留前向推理所需的 actor 网络”，而不是完整训练状态。

因此：

```text
不要做：cp policy.pt logs/.../model_1000.pt 然后 --resume
```

这不是可靠的恢复训练方式。

### 服务器端推荐工作流

建议按下面顺序继续：

1.  在训练工程中创建低墙任务环境。
2.  保持 actor 观测维度、动作维度、actor MLP 结构与 `RobotLab-Isaac-Velocity-Rough-RC-v0` 一致。
3.  启动一个新的 PPO 训练 run，而不是 resume 旧 run。
4.  在训练开始前，将部署导出的 `policy.pt` 加载为 actor 初始化权重。
5.  如果 `policy.pt` 内含 actor normalizer，则一并加载到当前策略的 actor normalizer。
6.  critic 保持随机初始化，让它在低墙任务中重新学习。

可以把它理解为：

```text
actor: 使用旧部署策略热启动
critic: 从零开始
optimizer: 从零开始
训练日志: 新建
```

这对于“已有全向爬楼梯/粗糙地形策略，现要迁移到低墙任务”的情况是合理且常用的做法。

### 适用条件

只有在下面条件满足时，才建议直接拿 `policy.pt` 做 actor 初始化：

*   该 `policy.pt` 确实来自 RC 的 `rough` 或兼容任务。
*   当前低墙任务的 actor 输入顺序与 `rough` 完全一致。
*   当前低墙任务的 actor 输入维度与 `rough` 完全一致。
*   当前低墙任务的动作维度与 `rough` 完全一致。
*   当前 PPO 配置中的 actor 网络结构与原策略一致，例如：

```text
actor_hidden_dims = [512, 256, 128]
activation = elu
```

如果你之后给 actor 新增了例如 `wall_height_obs`、`wall_distance_obs` 这类新观测，那么输入维度就变了，不能再直接完整加载该 `policy.pt` 作为 actor 初始化，除非额外做部分权重迁移。

### 服务器端需要准备的信息

在服务器上继续工作前，建议先整理以下内容：

*   `policy.pt` 的绝对路径。
*   该策略原始对应任务名。
*   是否确定它来自 `rsl_rl` 导出的部署策略。
*   当前服务器训练代码中 RC `rough` 的 actor hidden dims 和 activation。
*   当前低墙任务是否严格保持了与 `rough` 相同的 actor 观测与动作定义。

建议在服务器目录里单独记一个简短说明，例如：

```text
policy source: deployed RC rough policy
task source: RobotLab-Isaac-Velocity-Rough-RC-v0
obs layout: same as rough
action layout: same as rough
actor dims: [512, 256, 128]
```

这样后续不会混淆模型来源。

### 目录建议

建议在服务器训练工程中建立一个预训练权重目录，例如：

```text
artifacts/pretrained/rc_rough/policy.pt
```

或者：

```text
checkpoints/bootstrap/rc_rough_policy.pt
```

不要把它伪装成训练过程中生成的 `model_*.pt`。

### 训练脚本建议改法

推荐在训练脚本中增加一个新的命令行参数，例如：

```text
--pretrained_policy_jit /abs/path/to/policy.pt
```

其逻辑为：

1.  正常创建 env 和 PPO runner。
2.  在 `runner.learn(...)` 之前：
    *   `torch.jit.load(policy_path)`
    *   提取其中的 actor
    *   将权重加载到当前 runner 的 actor
    *   若存在 normalizer，则加载到 actor normalizer
3.  不加载旧 critic，不加载旧 optimizer，不设置 `--resume`

这样最清晰，也最不容易误用。

### 命令层面的工作方式

因为这是“初始化训练”而不是“恢复训练”，服务器端命令应当是“新训练命令 + 预训练 actor 路径”，而不是 `--resume`。

示意形式：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-LowWall-RC-v0 \
  --pretrained_policy_jit /abs/path/to/policy.pt \
  --run_name low_wall_from_deploy_policy
```

这里的关键点是：

*   不加 `--resume`
*   不依赖旧 `logs/rsl_rl/...` 中的 run 目录
*   这是一个全新的训练 run

### 如果后续找到了原始 `model_*.pt`

如果之后在旧训练环境、服务器日志或备份中找到了真正的训练 checkpoint，例如：

```text
model_1000.pt
```

那么优先使用真正的训练 checkpoint，因为它可以更完整地恢复：

*   actor
*   critic
*   optimizer 状态
*   训练进度

这时才推荐使用：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task <new_task> \
  --resume \
  --experiment_name rc_rough \
  --load_run <old_run_dir> \
  --checkpoint model_1000.pt
```

也就是说：

*   有 `model_*.pt` 时，优先 `--resume`
*   只有 `policy.pt` 时，使用“actor 初始化再新开训练”

### 针对低墙任务的具体建议

对于当前“跨越约 0.30 米低墙”任务，如果你准备先做第一版服务器实验，推荐：

1.  先保持 actor 观测完全与 `rough` 一致。
2.  暂时不要给 actor 添加 `wall_height_obs` 和 `wall_distance_obs`。
3.  使用部署导出的 `policy.pt` 初始化 actor。
4.  只训练“正面跨越、墙体居中、单墙、低高度 curriculum”的第一版。
5.  先验证从已有 locomotion policy 出发，是否比从零训练更快学会小墙跨越。

推荐初始课程：

```text
wall height: 0.05 -> 0.10 -> 0.15 m
wall orientation: perpendicular only
wall lateral offset: 0
command during crossing: vx > 0, vy = 0, wz = 0
```

如果这条线有效，再继续升级到更强版本，例如：

*   加墙高/墙距观测，
*   加 crossing mode，
*   做部分权重迁移而不是完整 actor 初始化。

### 服务器继续工作前的最短检查清单

开始前请确认：

*   `policy.pt` 是部署导出的 actor，而不是训练 checkpoint。
*   当前任务 actor 观测顺序与 `rough` 完全一致。
*   当前任务动作顺序与 `rough` 完全一致。
*   actor hidden dims 与 activation 一致。
*   将采用“新训练 + actor 初始化”，而不是 `--resume`。

如果以上都满足，那么这条工作流就是当前最稳妥的服务器端推进方式。

低门槛：

*   添加 `base_height_cmd` 。
*   添加基于命令的条件基础高度奖励。
*   增加低栏碰撞惩罚。
*   课程目标高度和横杆高度。

低墙：

*   增加墙面地形生成器。
*   增加障碍模式或步高指令。
*   增加墙壁进度和清除奖励。
*   课程墙壁高度从 0.05 米到 0.30 米。

石路：

*   增加石头/随机粗糙地形混合。
*   调整撞击、滑动和顺滑奖励。
*   调整课程凸起幅度和摩擦力。
*   在角色中无需高度扫描进行评估。
