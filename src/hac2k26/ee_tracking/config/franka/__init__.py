from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import franka_ee_tracking_env_cfg
from .rl_cfg import unitree_go1_getup_ppo_runner_cfg

register_mjlab_task(
    task_id="Hac2k26-EETracking-Franka",
    env_cfg=franka_ee_tracking_env_cfg(),
    play_env_cfg=franka_ee_tracking_env_cfg(play=True),
    rl_cfg=unitree_go1_getup_ppo_runner_cfg(),
)
