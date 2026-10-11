from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.rc.low_bar_env_cfg import (
    RCLowBarEnvCfg as QuadrupedRCLowBarEnvCfg,
)

from .common import RCWheeledActionsCfg, configure_wheeled_rc


@configclass
class RCLowBarEnvCfg(QuadrupedRCLowBarEnvCfg):
    actions: RCWheeledActionsCfg = RCWheeledActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_wheeled_rc(self, initial_height=0.28)
        self.commands.base_height.height_range = (0.18, 0.20)


@configclass
class RCLowBarPlayEnvCfg(RCLowBarEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
