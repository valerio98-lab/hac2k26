from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

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


def ee_ang_vel_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """End-effector angular vel in world frame."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    return asset.data.body_link_ang_vel_w[:, body_ids[0], :]


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


def traj_quat_ref(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Target EE quaternion (wxyz). Shape (N, 4)."""
    return env.command_manager.get_command(command_name)[:, 7:11]


def traj_omega_ref(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Target EE angular velocity (rad/s, world frame). Shape (N, 3)."""
    return env.command_manager.get_command(command_name)[:, 11:14]


def traj_lookahead(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Flattened future reference positions. Shape (N, 3 * lookahead_steps)."""
    return env.command_manager.get_command(command_name)[:, 14:]


def traj_rot6d_ref(
    env: ManagerBasedRlEnv, command_name: str = "trajectory"
) -> torch.Tensor:
    """Target EE orientation as 6D rotation (first two columns of R). Shape (N, 6)."""
    q_ref = env.command_manager.get_command(command_name)[:, 7:11]
    rotm = matrix_from_quat(q_ref)
    return _rot6d_from_matrix(rotm)


def pos_error_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "trajectory",
) -> torch.Tensor:
    """p_ref - p_ee in world frame. Shape (N, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    p_ee = asset.data.body_link_pos_w[:, body_ids[0], :]
    p_ref = env.command_manager.get_command(command_name)[:, 0:3]
    return p_ref - p_ee


def lin_vel_error_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "trajectory",
) -> torch.Tensor:
    """v_ref - v_ee in world frame. Shape (N, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    v_ee = asset.data.body_link_lin_vel_w[:, body_ids[0], :]
    v_ref = env.command_manager.get_command(command_name)[:, 3:6]
    return v_ref - v_ee


def rot_error_rotvec(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "trajectory",
) -> torch.Tensor:
    """Orientation error as world-frame axis-angle log(q_ref * q_ee^{-1}). Shape (N, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    q_ee = asset.data.body_link_quat_w[:, body_ids[0], :]
    q_ref = env.command_manager.get_command(command_name)[:, 7:11]
    return quat_box_minus(q_ref, q_ee)


def ang_vel_error_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "trajectory",
) -> torch.Tensor:
    """omega_ref - omega_ee in world frame. Shape (N, 3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    omega_ee = asset.data.body_link_ang_vel_w[:, body_ids[0], :]
    omega_ref = env.command_manager.get_command(command_name)[:, 11:14]
    return omega_ref - omega_ee
