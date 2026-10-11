from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.rc.broken_bridge_env_cfg import (
    RCBrokenBridgeEnvCfg as QuadrupedRCBrokenBridgeEnvCfg,
)

from .common import RCWheeledActionsCfg, configure_wheeled_rc


@configclass
class RCBrokenBridgeEnvCfg(QuadrupedRCBrokenBridgeEnvCfg):
    actions: RCWheeledActionsCfg = RCWheeledActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_wheeled_rc(self)


@configclass
class RCBrokenBridgePlayEnvCfg(RCBrokenBridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
