from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rot6d_from_matrix(rotm: torch.Tensor) -> torch.Tensor:
    """Encode rotation matrix as [c1(3), c2(3)] (first two columns)."""
    first_two_cols = rotm[..., :, :2]
    return first_two_cols.transpose(-2, -1).reshape(rotm.shape[0], -1)


def ee_pos_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """End-effector position in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    return asset.data.body_link_pos_w[:, body_ids[0], :]


def ee_lin_vel_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """End-effector vel in world frame"""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    return asset.data.body_link_lin_vel_w[:, body_ids[0], :]


def ee_rot6d_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """End-effector rot in world frame"""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    rotq = asset.data.body_link_quat_w[:, body_ids[0], :]
    rotm = matrix_from_quat(rotq)  # (N,..., 3, 3)
    return _rot6d_from_matrix(rotm)


def traj_pos_ref(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Current target EE position from the trajectory command. Shape (N, 3)."""
    return env.command_manager.get_command(command_name)[:, 0:3]


def traj_vel_ref(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Current target EE velocity from the trajectory command. Shape (N, 3)."""
    return env.command_manager.get_command(command_name)[:, 3:6]


def traj_phase(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Phase variable phi in [0, 1]. Shape (N, 1)."""
    return env.command_manager.get_command(command_name)[:, 6:7]


def traj_lookahead(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Flattened future reference positions. Shape (N, 3 * lookahead_steps)."""
    return env.command_manager.get_command(command_name)[:, 7:]
