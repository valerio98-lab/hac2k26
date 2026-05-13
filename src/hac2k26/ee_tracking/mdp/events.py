"""Reset events for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ...assets.robots.franka_panda.panda_constants import HOME_KEYFRAME

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_home(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset robots to home position"""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    J = len(HOME_KEYFRAME.joint_pos)
    asset: Entity = env.scene[asset_cfg.name]
    home_pos = torch.tensor(
        [i for i in HOME_KEYFRAME.joint_pos.values()], device=env.device
    )
    joint_home_pos = torch.zeros(size=(n, J + 2), device=env.device)
    joint_home_pos[..., :J] = home_pos
    joint_hom_vel = torch.zeros(size=(n, J + 2), device=env.device)

    asset.write_joint_state_to_sim(joint_home_pos, joint_hom_vel, env_ids=env_ids)

    env.sim.forward()
