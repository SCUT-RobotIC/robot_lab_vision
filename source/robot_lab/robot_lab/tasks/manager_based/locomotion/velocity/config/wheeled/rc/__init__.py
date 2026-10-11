import gymnasium as gym

from . import agents


for terrain, module, class_prefix in (
    ("Flat", "flat", "RCFlat"),
    ("Rough", "rough", "RCRough"),
    ("RoughStones", "rough_stones", "RCRoughStones"),
    ("LowBar", "low_bar", "RCLowBar"),
    ("LowWall", "low_wall", "RCLowWall"),
    ("BrokenBridge", "broken_bridge", "RCBrokenBridge"),
):
    for play in (False, True):
        suffix = "-Play" if play else ""
        cfg_suffix = "PlayEnvCfg" if play else "EnvCfg"
        gym.register(
            id=f"RobotLab-Isaac-Velocity-{terrain}{suffix}-RC27-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{module}_env_cfg:{class_prefix}{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{class_prefix}PPORunnerCfg",
            },
        )
