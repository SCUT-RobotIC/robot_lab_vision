#!/usr/bin/env python3
"""Keyboard/demo MuJoCo runner for the RC low-bar policy.

Keyboard mode:
  W/S: increase/decrease forward velocity
  A/D: increase/decrease lateral velocity
  Q/E: turn left/right
  Z/X: lower/raise base-height command
  Space: zero velocity command
  R: reset robot
  Esc: quit

Demo mode cycles through forward, backward, lateral and turning commands.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch

from sim2sim_rc_low_bar import (
    ACTION_SCALE,
    DECIMATION,
    DEFAULT_JOINT_POS,
    DEFAULT_XML,
    EFFORT_LIMIT,
    HISTORY,
    KD,
    KP,
    OBS_DIM,
    PHYSICS_DT,
    POLICY_JOINT_NAMES,
    TermHistory,
    load_policy,
    make_terms,
    read_policy_ordered_joints,
    resolve_model_indices,
    set_initial_state,
)


DEMO_COMMANDS = (
    ("forward", np.array([0.35, 0.0, 0.0], dtype=np.float32), 0.15),
    ("slow forward", np.array([0.18, 0.0, 0.0], dtype=np.float32), 0.15),
    ("left strafe", np.array([0.20, 0.12, 0.0], dtype=np.float32), 0.15),
    ("right strafe", np.array([0.20, -0.12, 0.0], dtype=np.float32), 0.15),
    ("turn left", np.array([0.20, 0.0, 0.25], dtype=np.float32), 0.15),
    ("turn right", np.array([0.20, 0.0, -0.25], dtype=np.float32), 0.15),
    ("low crawl", np.array([0.25, 0.0, 0.0], dtype=np.float32), 0.13),
)


class CommandState:
    def __init__(self, vx: float, vy: float, yaw_rate: float, base_height: float):
        self.command = np.array([vx, vy, yaw_rate], dtype=np.float32)
        self.base_height = float(base_height)
        self.quit = False
        self.reset_requested = False
        self.demo_name = "manual"

    def clamp(self) -> None:
        self.command[0] = float(np.clip(self.command[0], -0.35, 0.60))
        self.command[1] = float(np.clip(self.command[1], -0.25, 0.25))
        self.command[2] = float(np.clip(self.command[2], -0.50, 0.50))
        self.base_height = float(np.clip(self.base_height, 0.12, 0.20))

    def zero_velocity(self) -> None:
        self.command[:] = 0.0

    def status(self) -> str:
        return (
            f"{self.demo_name}: vx={self.command[0]: .2f}, vy={self.command[1]: .2f}, "
            f"yaw={self.command[2]: .2f}, h={self.base_height: .2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard/demo MuJoCo sim2sim for RobotLab RC low-bar policy.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to the MuJoCo XML.")
    parser.add_argument("--policy", type=Path, required=True, help="Path to exported TorchScript policy.pt.")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds.")
    parser.add_argument("--vx", type=float, default=0.35, help="Initial commanded x velocity.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial commanded y velocity.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Initial commanded yaw rate.")
    parser.add_argument("--base-height", type=float, default=0.15, help="Initial commanded base height.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--action-delay", type=int, default=4, help="Fixed action delay in 200 Hz physics steps.")
    parser.add_argument("--real-time", action="store_true", default=True, help="Sleep to approximately real time.")
    parser.add_argument("--no-real-time", action="store_false", dest="real_time", help="Run as fast as possible.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--demo", action="store_true", help="Cycle through a multi-direction command demo.")
    parser.add_argument("--demo-period", type=float, default=4.0, help="Seconds per demo command.")
    parser.add_argument("--print-rate", type=float, default=0.5, help="Status print period in seconds. Use 0 to disable.")
    return parser.parse_args()


def print_keyboard_help() -> None:
    print(
        "Keyboard: W/S vx, A/D vy, Q/E yaw, Z/X height, Space stop, R reset, Esc quit. "
        "Click the MuJoCo window first if keys do not register."
    )


def make_key_callback(state: CommandState):
    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            key = ""

        if key == "w":
            state.command[0] += 0.05
        elif key == "s":
            state.command[0] -= 0.05
        elif key == "a":
            state.command[1] += 0.05
        elif key == "d":
            state.command[1] -= 0.05
        elif key == "q":
            state.command[2] += 0.05
        elif key == "e":
            state.command[2] -= 0.05
        elif key == "z":
            state.base_height -= 0.01
        elif key == "x":
            state.base_height += 0.01
        elif key == " ":
            state.zero_velocity()
        elif key == "r":
            state.reset_requested = True
        elif keycode == 256:
            state.quit = True
        state.demo_name = "manual"
        state.clamp()
        print(state.status())

    return key_callback


def make_history(data, qpos_ids, qvel_ids, state: CommandState, last_action: np.ndarray) -> TermHistory:
    history = TermHistory(
        {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos": 12,
            "joint_vel": 12,
            "actions": 12,
            "base_height_command": 1,
        },
        HISTORY,
    )
    terms = make_terms(data, qpos_ids, qvel_ids, state.command, state.base_height, last_action)
    history.fill(terms)
    return history


def apply_demo_command(state: CommandState, sim_time: float, demo_period: float) -> None:
    index = int(sim_time / demo_period) % len(DEMO_COMMANDS)
    name, command, base_height = DEMO_COMMANDS[index]
    state.demo_name = name
    state.command[:] = command
    state.base_height = base_height
    state.clamp()


def main() -> None:
    args = parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("Missing dependency: activate isaac_vision or install mujoco with `pip install mujoco`.") from exc

    viewer_mod = None
    if not args.headless:
        try:
            import mujoco.viewer as viewer_mod
        except ImportError as exc:
            raise SystemExit("MuJoCo viewer is unavailable. Re-run with --headless or install viewer dependencies.") from exc

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if abs(model.opt.timestep - PHYSICS_DT) > 1.0e-9:
        raise RuntimeError(f"MJCF timestep must be {PHYSICS_DT}, got {model.opt.timestep}.")

    qpos_ids, qvel_ids, actuator_ids, actuator_joint_names = resolve_model_indices(model)
    set_initial_state(data, qpos_ids, qvel_ids)
    mujoco.mj_forward(model, data)

    state = CommandState(args.vx, args.vy, args.yaw_rate, args.base_height)
    state.clamp()
    policy = load_policy(args.policy, args.device)
    last_action = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
    action_scale = np.array([ACTION_SCALE[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    default_q = np.array([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    kp = np.array([KP[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    kd = np.array([KD[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    effort_limit = np.array([EFFORT_LIMIT[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)

    history = make_history(data, qpos_ids, qvel_ids, state, last_action)
    delay_steps = max(args.action_delay, 0)
    delayed_actions = collections.deque([last_action.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
    target_q = default_q.copy()
    total_steps = int(args.duration / PHYSICS_DT)
    next_print_time = 0.0
    last_demo_index = -1

    viewer = None
    if viewer_mod is not None:
        print_keyboard_help()
        viewer = viewer_mod.launch_passive(model, data, key_callback=make_key_callback(state))

    try:
        for step in range(total_steps):
            if state.quit:
                break

            loop_start = time.time()

            if args.demo:
                demo_index = int(data.time / args.demo_period) % len(DEMO_COMMANDS)
                apply_demo_command(state, data.time, args.demo_period)
                if demo_index != last_demo_index:
                    print(state.status())
                    last_demo_index = demo_index

            if state.reset_requested:
                set_initial_state(data, qpos_ids, qvel_ids)
                mujoco.mj_forward(model, data)
                last_action[:] = 0.0
                delayed_actions = collections.deque(
                    [last_action.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1
                )
                target_q = default_q.copy()
                history = make_history(data, qpos_ids, qvel_ids, state, last_action)
                state.reset_requested = False

            if step % DECIMATION == 0:
                terms = make_terms(data, qpos_ids, qvel_ids, state.command, state.base_height, last_action)
                history.append(terms)
                obs_np = history.flatten()
                if obs_np.shape != (OBS_DIM,):
                    raise RuntimeError(f"Expected obs shape ({OBS_DIM},), got {obs_np.shape}.")
                obs = torch.from_numpy(obs_np).unsqueeze(0).to(args.device)
                with torch.inference_mode():
                    action = policy(obs)
                last_action = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
                if last_action.shape != (len(POLICY_JOINT_NAMES),):
                    raise RuntimeError(f"Expected action shape ({len(POLICY_JOINT_NAMES)},), got {last_action.shape}.")
                last_action = np.clip(last_action, -100.0, 100.0)

            delayed_actions.append(last_action.copy())
            applied_action = delayed_actions[0]
            target_q = default_q + applied_action * action_scale

            q, qd = read_policy_ordered_joints(data, qpos_ids, qvel_ids)
            tau_policy = kp * (target_q - q) - kd * qd
            tau_policy = np.clip(tau_policy, -effort_limit, effort_limit)

            for actuator_name, aid in actuator_ids.items():
                joint_name = actuator_joint_names[actuator_name]
                policy_index = POLICY_JOINT_NAMES.index(joint_name)
                data.ctrl[aid] = tau_policy[policy_index]

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            if args.print_rate > 0 and data.time >= next_print_time:
                base_pos = data.qpos[:3]
                print(
                    f"t={data.time:5.2f}s base=({base_pos[0]: .2f}, {base_pos[1]: .2f}, {base_pos[2]: .2f}) "
                    f"{state.status()}"
                )
                next_print_time += args.print_rate

            if args.real_time:
                elapsed = time.time() - loop_start
                sleep_time = PHYSICS_DT - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
