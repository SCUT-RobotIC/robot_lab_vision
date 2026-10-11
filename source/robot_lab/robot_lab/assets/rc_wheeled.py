"""27dog wheel-legged asset and M3508 output-shaft limits."""

import math

from isaaclab.actuators import ImplicitActuatorCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.rc import RC_CFG

WHEEL_RATED_EFFORT = 2.46
WHEEL_MAX_EFFORT = 3.69
WHEEL_RATED_VELOCITY = 571.0 * 2.0 * math.pi / 60.0
WHEEL_MAX_VELOCITY = 587.0 * 2.0 * math.pi / 60.0
WHEEL_RADIUS = 0.06
RC27_INITIAL_HEIGHT = 0.48
RC27_STANDING_HEIGHT = 0.42

RC_WHEELED_CFG = RC_CFG.copy()
RC_WHEELED_CFG.spawn.asset_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/RC/27dog urdf/urdf/27dog.SLDASM.urdf"
RC_WHEELED_CFG.spawn.force_usd_conversion = True
RC_WHEELED_CFG.init_state.pos = (0.0, 0.0, RC27_INITIAL_HEIGHT)
RC_WHEELED_CFG.init_state.joint_pos[".*_feet_joint"] = 0.0
RC_WHEELED_CFG.actuators["wheels"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_feet_joint"],
    effort_limit_sim=WHEEL_MAX_EFFORT,
    velocity_limit_sim=WHEEL_MAX_VELOCITY,
    stiffness=0.0,
    damping=0.5,
    friction=0.0,
    armature=0.001,
)
