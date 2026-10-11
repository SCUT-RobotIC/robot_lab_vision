from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.rc.low_wall_env_cfg import (
    RCLowWallEnvCfg as QuadrupedRCLowWallEnvCfg,
)

from .common import RCWheeledActionsCfg, configure_wheeled_rc


@configclass
class RCLowWallEnvCfg(QuadrupedRCLowWallEnvCfg):
    actions: RCWheeledActionsCfg = RCWheeledActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_wheeled_rc(self)


@configclass
class RCLowWallPlayEnvCfg(RCLowWallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.75, 0.75)
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
