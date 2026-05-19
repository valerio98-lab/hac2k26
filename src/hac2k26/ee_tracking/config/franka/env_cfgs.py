from hac2k26.assets.robots import get_panda_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from hac2k26.ee_tracking.tracking_env_cfg import make_ee_tracking_env_cfg


def _apply_franka_overrides(
    cfg: ManagerBasedRlEnvCfg, play: bool
) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities = {"robot": get_panda_robot_cfg()}
    cfg.viewer.body_name = "link0"
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def franka_ee_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Franka EE tracking with random quintic-spline waypoint trajectories."""
    cfg = make_ee_tracking_env_cfg()
    return _apply_franka_overrides(cfg, play)
