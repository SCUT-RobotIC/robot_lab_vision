#!/usr/bin/env python3
"""Run the RC low-bar policy in MuJoCo.

This script mirrors the IsaacLab RCLowBar policy interface:
- 200 Hz physics / PD loop.
- 50 Hz policy loop.
- explicit MuJoCo-to-policy joint remap.
- 230-D policy observation with term-wise 5-frame history.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "source/robot_lab/data/Robots/RC/DOG/mjcf/rc_low_bar.xml"

POLICY_JOINT_NAMES = [
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

MUJOCO_ACTUATOR_NAMES = [
    "FL_hip_motor",
    "FL_thigh_motor",
    "FL_calf_motor",
    "FR_hip_motor",
    "FR_thigh_motor",
    "FR_calf_motor",
    "RL_hip_motor",
    "RL_thigh_motor",
    "RL_calf_motor",
    "RR_hip_motor",
    "RR_thigh_motor",
    "RR_calf_motor",
]

DEFAULT_JOINT_POS = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 1.10,
    "FR_calf_joint": -2.45,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 1.10,
    "FL_calf_joint": -2.45,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.40,
    "RR_calf_joint": -2.45,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 1.40,
    "RL_calf_joint": -2.45,
}

ACTION_SCALE = {
    "FR_hip_joint": 0.125,
    "FR_thigh_joint": 0.25,
    "FR_calf_joint": 0.25,
    "FL_hip_joint": 0.125,
    "FL_thigh_joint": 0.25,
    "FL_calf_joint": 0.25,
    "RR_hip_joint": 0.125,
    "RR_thigh_joint": 0.25,
    "RR_calf_joint": 0.25,
    "RL_hip_joint": 0.125,
    "RL_thigh_joint": 0.25,
    "RL_calf_joint": 0.25,
}

KP = {
    "FR_hip_joint": 25.0,
    "FR_thigh_joint": 25.0,
    "FR_calf_joint": 25.0,
    "FL_hip_joint": 25.0,
    "FL_thigh_joint": 25.0,
    "FL_calf_joint": 25.0,
    "RR_hip_joint": 25.0,
    "RR_thigh_joint": 25.0,
    "RR_calf_joint": 25.0,
    "RL_hip_joint": 25.0,
    "RL_thigh_joint": 25.0,
    "RL_calf_joint": 25.0,
}

KD = {
    "FR_hip_joint": 2.0,
    "FR_thigh_joint": 2.0,
    "FR_calf_joint": 2.0,
    "FL_hip_joint": 2.0,
    "FL_thigh_joint": 2.0,
    "FL_calf_joint": 2.0,
    "RR_hip_joint": 2.0,
    "RR_thigh_joint": 2.0,
    "RR_calf_joint": 2.0,
    "RL_hip_joint": 2.0,
    "RL_thigh_joint": 2.0,
    "RL_calf_joint": 2.0,
}

EFFORT_LIMIT = {
    "FR_hip_joint": 20.0,
    "FR_thigh_joint": 30.0,
    "FR_calf_joint": 30.0,
    "FL_hip_joint": 20.0,
    "FL_thigh_joint": 30.0,
    "FL_calf_joint": 30.0,
    "RR_hip_joint": 20.0,
    "RR_thigh_joint": 30.0,
    "RR_calf_joint": 30.0,
    "RL_hip_joint": 20.0,
    "RL_thigh_joint": 30.0,
    "RL_calf_joint": 30.0,
}

OBS_DIM = 230
HISTORY = 5
PHYSICS_DT = 0.005
DECIMATION = 4


class TermHistory:
    """Term-wise history that matches IsaacLab observation term flattening."""

    def __init__(self, shapes: dict[str, int], history: int):
        self._buffers = {
            name: collections.deque([np.zeros(size, dtype=np.float32) for _ in range(history)], maxlen=history)
            for name, size in shapes.items()
        }

    def append(self, terms: dict[str, np.ndarray]) -> None:
        for name, value in terms.items():
            self._buffers[name].append(np.asarray(value, dtype=np.float32).copy())

    def fill(self, terms: dict[str, np.ndarray]) -> None:
        for _ in range(HISTORY):
            self.append(terms)

    def flatten(self) -> np.ndarray:
        pieces = []
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "base_height_command",
        ):
            pieces.append(np.concatenate(list(self._buffers[name]), axis=0))
        obs = np.concatenate(pieces, axis=0).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"Expected obs shape ({OBS_DIM},), got {obs.shape}.")
        return obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim for RobotLab RC low-bar policy.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to the MuJoCo XML.")
    parser.add_argument("--policy", type=Path, required=True, help="Path to exported TorchScript policy.pt.")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--vx", type=float, default=0.35, help="Commanded x velocity.")
    parser.add_argument("--vy", type=float, default=0.0, help="Commanded y velocity.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Commanded yaw rate.")
    parser.add_argument("--base-height", type=float, default=0.15, help="Commanded base height.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--real-time", action="store_true", help="Sleep to approximately real time.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--action-delay", type=int, default=4, help="Fixed action delay in 200 Hz physics steps.")
    parser.add_argument("--print-rate", type=float, default=1.0, help="Status print period in seconds. Use 0 to disable.")
    return parser.parse_args()


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def get_projected_gravity(data) -> np.ndarray:
    rot_world_from_body = quat_to_rotmat_wxyz(data.qpos[3:7])
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return rot_world_from_body.T @ gravity_world


def get_base_ang_vel_body(data) -> np.ndarray:
    rot_world_from_body = quat_to_rotmat_wxyz(data.qpos[3:7])
    return rot_world_from_body.T @ data.qvel[3:6]


def resolve_model_indices(model):
    mujoco_joint_qpos = {}
    mujoco_joint_qvel = {}
    for name in POLICY_JOINT_NAMES:
        jid = model.joint(name).id
        mujoco_joint_qpos[name] = model.jnt_qposadr[jid]
        mujoco_joint_qvel[name] = model.jnt_dofadr[jid]

    actuator_ids = {}
    actuator_joint_names = {}
    for actuator_name in MUJOCO_ACTUATOR_NAMES:
        aid = model.actuator(actuator_name).id
        actuator_ids[actuator_name] = aid
        actuator_joint_names[actuator_name] = actuator_name[: -len("_motor")] + "_joint"

    return mujoco_joint_qpos, mujoco_joint_qvel, actuator_ids, actuator_joint_names


def read_policy_ordered_joints(data, qpos_ids: dict[str, int], qvel_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    q = np.array([data.qpos[qpos_ids[name]] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    qd = np.array([data.qvel[qvel_ids[name]] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    return q, qd


def make_terms(
    data,
    qpos_ids: dict[str, int],
    qvel_ids: dict[str, int],
    command: np.ndarray,
    base_height_command: float,
    last_action: np.ndarray,
) -> dict[str, np.ndarray]:
    q, qd = read_policy_ordered_joints(data, qpos_ids, qvel_ids)
    default_q = np.array([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    return {
        "base_ang_vel": (get_base_ang_vel_body(data) * 0.25).astype(np.float32),
        "projected_gravity": get_projected_gravity(data).astype(np.float32),
        "velocity_commands": command.astype(np.float32),
        "joint_pos": (q - default_q).astype(np.float32),
        "joint_vel": (qd * 0.05).astype(np.float32),
        "actions": last_action.astype(np.float32),
        "base_height_command": np.array([base_height_command], dtype=np.float32),
    }


def set_initial_state(data, qpos_ids: dict[str, int], qvel_ids: dict[str, int]) -> None:
    data.qpos[:7] = np.array([0.0, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[:6] = 0.0
    for name, value in DEFAULT_JOINT_POS.items():
        data.qpos[qpos_ids[name]] = value
        data.qvel[qvel_ids[name]] = 0.0


def load_policy(path: Path, device: str):
    policy = torch.jit.load(str(path), map_location=device)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("Missing dependency: install mujoco with `pip install mujoco`.") from exc

    viewer = None
    if not args.headless:
        try:
            import mujoco.viewer
        except ImportError as exc:
            raise SystemExit("MuJoCo viewer is unavailable. Re-run with --headless or install viewer dependencies.") from exc

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if abs(model.opt.timestep - PHYSICS_DT) > 1.0e-9:
        raise RuntimeError(f"MJCF timestep must be {PHYSICS_DT}, got {model.opt.timestep}.")

    qpos_ids, qvel_ids, actuator_ids, actuator_joint_names = resolve_model_indices(model)
    set_initial_state(data, qpos_ids, qvel_ids)
    mujoco.mj_forward(model, data)

    policy = load_policy(args.policy, args.device)
    command = np.array([args.vx, args.vy, args.yaw_rate], dtype=np.float32)
    last_action = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
    action_scale = np.array([ACTION_SCALE[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    default_q = np.array([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    kp = np.array([KP[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    kd = np.array([KD[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    effort_limit = np.array([EFFORT_LIMIT[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)

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
    initial_terms = make_terms(data, qpos_ids, qvel_ids, command, args.base_height, last_action)
    history.fill(initial_terms)

    delay_steps = max(args.action_delay, 0)
    delayed_actions = collections.deque([last_action.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)
    target_q = default_q.copy()
    total_steps = int(args.duration / PHYSICS_DT)
    next_print_time = 0.0

    if viewer is None and not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        for step in range(total_steps):
            loop_start = time.time()

            if step % DECIMATION == 0:
                terms = make_terms(data, qpos_ids, qvel_ids, command, args.base_height, last_action)
                history.append(terms)
                obs = torch.from_numpy(history.flatten()).unsqueeze(0).to(args.device)
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
                    f"|action|={np.linalg.norm(last_action):.2f} |tau|={np.linalg.norm(tau_policy):.2f}"
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
