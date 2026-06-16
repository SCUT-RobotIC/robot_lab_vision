# RC New Locomotion Task Training Plan

This note records proposed training plans for three new RC robot tasks:

1. body-height command for ducking under a low bar,
2. climbing over low walls around 0.30 m,
3. walking over uneven stone-like ground.

The robot already has flat, slope, and stair walking policies. The recommended
starting point is the velocity RC stack:

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/rc/rough_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/rc/agents/rsl_rl_ppo_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

For real-robot transfer, keep the deployable actor observation limited to
signals available on hardware: angular velocity, projected gravity, velocity
command, joint position, joint velocity, and previous actions. Keep privileged
information such as terrain height scan, true base linear velocity, contacts,
terrain type, and torques in the critic or teacher only.

## Common Strategy

Use the existing trained walking policy as initialization whenever possible.

Recommended progression:

1. Train or fine-tune with PPO using asymmetric actor-critic.
2. Actor observes only deployable observations.
3. Critic observes privileged simulation signals.
4. If the privileged critic is not enough, train a privileged teacher and
   distill into a deployable MLP student.
5. Export only the actor/student for deployment.

Do not mix all three new tasks at full difficulty at the beginning. Train three
specialized policies first, then optionally merge them through curriculum or
multi-task commands.

## Task 1: Body-Height Command for Low-Bar Ducking

Goal: add a commanded body height so the robot can lower its body and pass under
a height-limited bar while still tracking velocity.

Important competition constraint: the bar is not fixed. If the robot touches and
knocks the bar down, this obstacle fails. Therefore this task should be trained
as a strict no-contact clearance task, not as a task where the robot may scrape
against a fixed collision object.

### Command Design

Extend the velocity command from `[vx, vy, yaw]` to include target body height:

```text
[vx, vy, yaw, base_height_cmd]
```

Use a conservative command range first:

```text
base_height_cmd: 0.22 m to 0.35 m
normal standing: about 0.30 m to 0.35 m
ducking phase: about 0.22 m to 0.28 m
```

Implementation options:

- Add a new command generator in `velocity/mdp/commands.py`, extending
  `UniformThresholdVelocityCommand`.
- Or add a second command term named `base_height` and concatenate it into
  observations through `mdp.generated_commands`.

The second option is cleaner if the existing velocity command should remain
compatible with old policies.

### Environment Design

Start with a simple low-bar scene before randomizing heavily.

Training phases:

1. No physical bar, only height command tracking.
2. Add a non-moving reference bar with generous clearance and terminate on any
   robot-bar contact.
3. Replace or validate with a light dynamic bar that can be displaced.
4. Randomize bar height, width, radius, and position.
5. Mix normal walking and ducking episodes.

Bar geometry should not be part of actor observation unless the real robot has
a matching perception signal. If the real task is command-driven, the operator
or upstream planner should command the ducking height before the obstacle.

Implementation note: a static bar with contact termination is a good first
training approximation because it gives a clean failure signal. Before trusting
the policy, validate it with a light dynamic bar and terminate when either
contact occurs or the bar displacement exceeds a small threshold. This avoids a
policy that learns to brush the bar lightly in simulation.

### Rewards

Add or reuse a base height tracking reward:

```text
r_height = exp(-(base_z - base_height_cmd)^2 / sigma^2)
```

Use the local terrain-relative base height when possible. Existing
`base_height_l2` can be adapted from a fixed target to a command target.

Recommended reward changes:

- Positive: body height tracks `base_height_cmd`.
- Positive: velocity tracking remains active but slightly downweighted during
  low-height command.
- Positive: clearance margin below the bar, if bar pose is available as
  privileged training information.
- Negative: pitch/roll too large.
- Negative: knee/hip joint limit proximity.
- Negative or termination: any robot contact with the bar.
- Termination: bar displacement or rotation exceeds the allowed threshold.
- Negative: large action rate and joint acceleration.

Avoid over-penalizing joint deviation during ducking. A low body posture
requires large leg configuration changes, so `joint_pos_penalty` and
`stand_still` may need command-dependent weights.

Do not rely only on a soft negative reward for bar contact. Since touching the
bar fails the obstacle in competition, contact should end the episode or at
least produce a very large penalty. A useful pattern is:

```text
success reward = pass the bar plane + no bar contact + no bar displacement
failure termination = any bar contact, or bar displacement > threshold
```

### Curriculum

Suggested curriculum:

1. `base_height_cmd` range: 0.30 m to 0.35 m.
2. Expand to 0.26 m to 0.35 m.
3. Expand to 0.22 m to 0.35 m.
4. Add no-contact low bar with height 0.38 m to 0.45 m.
5. Require a larger safety margin first, for example 5 cm to 8 cm below the bar.
6. Reduce clearance until it matches the target real low-bar task.
7. Validate with a light dynamic bar and strict displacement termination.

Success metrics:

- passes under bar with zero robot-bar contact,
- bar displacement stays below the competition-safe threshold,
- tracks commanded forward velocity,
- returns to normal height after the low section,
- no knee/hip saturation spikes.

### Real-Robot Notes

This is the most feasible of the three tasks without a depth camera, because
the obstacle can be handled by command scheduling. The policy does not need to
detect the bar if the operator or planner tells it when to lower.

Because the real bar can fall, deploy with a conservative height command. The
robot should duck slightly lower than the nominal measured clearance, leaving
margin for state-estimation error, body oscillation, actuator delay, and small
terrain height variation.

## Task 2: Climbing Over Low Walls Around 0.30 m

Goal: make the robot climb over many low walls around normal standing height.

This is the hardest task without terrain perception. A 0.30 m wall is not just
rough walking; it is an obstacle negotiation task requiring foot clearance,
body pitch management, and reliable contact transitions.

### Terrain Design

Create a new wall terrain generator or custom mesh terrain.

Wall randomization:

```text
height: 0.10 m -> 0.30 m curriculum
thickness: 0.08 m -> 0.25 m
top shape: flat first, then rounded/chamfered
spacing: 1.0 m -> 3.0 m
orientation: mostly perpendicular first, then small yaw variation
```

Start with single wall per terrain tile. Then increase to multiple walls per
tile only after the policy reliably clears one wall.

### Observation Design

For simulation training, use privileged critic or teacher observations:

- terrain height scan,
- foot contact state,
- base height,
- local terrain type or wall distance if available.

For deployable actor/student without depth camera:

- velocity command,
- IMU angular velocity,
- projected gravity,
- joint positions and velocities,
- previous actions,
- optionally foot contact sensors if the real robot has them.

If there is no perception and no wall timing command, the final real policy will
not know when to prepare for a wall. In that case it can only learn a slow,
reactive, high-clearance gait. This may work for scattered low obstacles but is
less reliable for true vertical walls.

Recommended command addition:

```text
obstacle_mode or step_height_cmd
```

A simple discrete command is enough:

```text
0 = normal walking
1 = obstacle crossing / high-step gait
```

This is more realistic than expecting an MLP with no perception to infer a wall
before impact.

### Rewards

Keep velocity tracking, but reduce its priority near walls. Add obstacle-specific
terms:

- Positive: forward progress across the wall.
- Positive: feet clear obstacle height during swing.
- Positive: base remains above a minimum height while crossing.
- Positive: all feet eventually reach the far side.
- Negative: body collision with wall.
- Negative: excessive contact force.
- Negative: foot dragging on wall face.
- Negative: roll/pitch beyond safe thresholds.
- Negative: high joint power and action jerk.

Existing useful rewards:

- `feet_air_time`
- `feet_height`
- `feet_height_body`
- `contact_forces`
- `action_rate_l2`
- `joint_power`
- `flat_orientation_l2`

Likely needed new rewards:

- `wall_clearance_feet`
- `wall_progress`
- `body_wall_collision`
- `front_feet_far_side`
- `rear_feet_far_side`

### Curriculum

Suggested curriculum:

1. Start from existing stair/slope policy.
2. Add low blocks at 0.05 m to 0.10 m.
3. Increase to 0.15 m.
4. Increase to 0.20 m with wider top surface.
5. Increase to 0.25 m.
6. Try 0.30 m only after the gait is stable.

Also curriculum the command speed:

```text
early: vx = 0.1 m/s to 0.3 m/s
later: vx = 0.2 m/s to 0.6 m/s
```

Avoid high lateral velocity early.

### Teacher-Student Option

This task is a good candidate for teacher-student training:

```text
teacher: MLP with height_scan / wall distance / terrain type / contact
student: MLP with deployable proprioception and optional obstacle_mode command
```

If the real robot has no perception, include an obstacle command in the student.
Otherwise the student has no way to know that a wall is coming.

### Real-Robot Notes

Treat 0.30 m as aggressive. Before real trials:

- validate maximum foot clearance in open loop,
- check knee and hip torque margin,
- add emergency body-contact termination in sim,
- start real tests with 0.05 m to 0.10 m obstacles,
- use a safety gantry or soft foam obstacle.

## Task 3: Walking Over Uneven Stone-Like Ground

Goal: make the robot walk over random bumps/depressions similar to a stone road.

This is the best fit for the current rough-terrain framework. It does not
require precise obstacle timing, and a proprioceptive policy can learn robust
reactions if the terrain is not too discontinuous.

### Terrain Design

Use or extend `HfRandomUniformTerrainCfg`.

Initial terrain ranges:

```text
noise_range: 0.01 m to 0.05 m
noise_step: 0.01 m to 0.02 m
horizontal_scale: 0.05 m to 0.10 m
```

Hard terrain ranges:

```text
noise_range: 0.05 m to 0.12 m
noise_step: 0.02 m to 0.04 m
```

For stone-like terrain, prefer a mix:

- random uniform heightfield,
- small discrete steps,
- low pyramidal bumps,
- sparse raised patches,
- friction variation.

Avoid starting with extremely sharp height discontinuities; they often produce
unrealistic contact impulses and poor sim-to-real transfer.

### Observation Design

Actor can remain deployable:

- no height scan,
- no true base linear velocity unless available from state estimation,
- keep IMU, joint states, commands, previous actions.

Critic can keep:

- height scan,
- base linear velocity,
- base position z,
- contact forces,
- joint effort.

This is exactly the asymmetric actor-critic pattern already present in the RC
velocity stack.

### Rewards

Reuse most of the current rough walking rewards, with these adjustments:

- Increase robustness terms gradually, not all at once.
- Keep `track_lin_vel_xy_exp` and `track_ang_vel_z_exp`.
- Keep moderate `base_height_l2`, but do not force a perfectly fixed height on
  rough terrain.
- Keep `flat_orientation_l2`, `ang_vel_xy_l2`, and `lin_vel_z_l2`.
- Increase `contact_forces` penalty if impacts are violent.
- Keep `feet_slide` if foot slipping becomes visible.
- Keep `action_rate_l2` and `joint_acc_l2` for smoother real behavior.

Potential new reward:

- `foot_clearance_adaptive`: reward swing foot clearance relative to local
  terrain height, using privileged terrain data only for reward.

### Curriculum

Suggested curriculum:

1. Mix 70 percent plane/slope/stair, 30 percent mild random rough.
2. Increase random rough proportion to 50 percent.
3. Increase bump amplitude.
4. Add friction randomization.
5. Add push disturbances.
6. Reduce terrain cache determinism so the policy does not memorize patterns.

For real deployment, this should probably be trained as an extension of the
existing rough policy rather than from scratch.

### Real-Robot Notes

This task is the most realistic without a depth camera. The robot can react
through IMU and joint/contact dynamics. If foot contact sensors are available,
adding binary foot contact to actor observations may significantly improve
performance on loose or uneven surfaces.

## Recommended Priority

1. Stone-like uneven ground.
   Lowest perception requirement and best match to existing rough locomotion.

2. Body-height command for low-bar ducking.
   Feasible if the low posture is commanded by an operator or planner.

3. Low wall climbing.
   Highest risk. Strongly recommended to use either perception or an explicit
   obstacle-crossing command. Use teacher-student only if the teacher has
   privileged wall/height information and the student gets at least a mode or
   timing command.

## Implementation Checklist

Shared:

- Keep actor observation deployable.
- Keep critic privileged.
- Fine-tune from existing RC rough/flat checkpoints.
- Add one task at a time.
- Export and validate observation/action order before real deployment.

Low-bar:

- Add `base_height_cmd`.
- Add command-conditioned base-height reward.
- Add low-bar collision penalty.
- Curriculum target height and bar clearance.

Low-wall:

- Add wall terrain generator.
- Add obstacle mode or step-height command.
- Add wall progress and clearance rewards.
- Curriculum wall height from 0.05 m to 0.30 m.

Stone-road:

- Add stone/random rough terrain mix.
- Tune impact, slip, and smoothness rewards.
- Curriculum bump amplitude and friction.
- Evaluate with no height scan in actor.
