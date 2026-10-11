from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.rc.agents import rsl_rl_ppo_cfg as quadruped


@configclass
class RCFlatPPORunnerCfg(quadruped.RCFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_flat"


@configclass
class RCRoughPPORunnerCfg(quadruped.RCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_rough"


@configclass
class RCRoughStonesPPORunnerCfg(quadruped.RCRoughStonesPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_rough_stones"


@configclass
class RCLowBarPPORunnerCfg(quadruped.RCLowBarPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_low_bar"


@configclass
class RCLowWallPPORunnerCfg(quadruped.RCLowWallPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_low_wall"
        self.load_experiment_name = "rc27_rough_stones"
        self.load_run = ".*"
        self.load_checkpoint = "model_.*.pt"


@configclass
class RCBrokenBridgePPORunnerCfg(quadruped.RCBrokenBridgePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "rc27_broken_bridge"
        self.load_experiment_name = "rc27_rough_stones"
        self.load_run = ".*"
        self.load_checkpoint = "model_.*.pt"
