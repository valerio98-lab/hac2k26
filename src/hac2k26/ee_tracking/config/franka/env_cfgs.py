from hac2k26.assets.robots import get_panda_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from hac2k26.ee_tracking.getup_env_cfg import make_ee_tracking_env_cfg


def franka_ee_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Franka EE tracking task configuration."""
    cfg = make_ee_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_panda_robot_cfg()}
    cfg.viewer.body_name = "link0"

    if play:
        cfg.observations["actor"].enable_corruption = False

    return cfg
