"""Reward functions for the EE tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_idx(asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    assert isinstance(body_ids, list)
    return body_ids[0]


def pos_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    alpha: float = 50.0,
) -> torch.Tensor:
    """Gaussian kernel on EE position error: exp(-alpha * ||p_ee - p_ref||^2)."""
    asset: Entity = env.scene[asset_cfg.name]
    p_ee = asset.data.body_link_pos_w[:, _body_idx(asset_cfg), :]
    p_ref = env.command_manager.get_command("trajectory")[:, 0:3]
    err_sq = torch.sum(torch.square(p_ee - p_ref), dim=-1)
    return torch.exp(-alpha * err_sq)


def orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    beta: float = 5.0,
) -> torch.Tensor:
    """Gaussian kernel on geodesic angle error: exp(-beta * theta_err^2)."""
    asset: Entity = env.scene[asset_cfg.name]
    q_ee = asset.data.body_link_quat_w[:, _body_idx(asset_cfg), :]
    q_ref = env.command_manager.get_command("trajectory")[:, 7:11]
    theta_err = quat_error_magnitude(q_ee, q_ref)
    return torch.exp(-beta * theta_err**2)


def velocity_alignment_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Cosine alignment between EE velocity and the trajectory tangent.

    r = v_ee . t_hat,  t_hat = v_ref / (||v_ref|| + eps)

    Naturally vanishes near waypoints where ||v_ref|| -> 0 (rest-to-rest),
    so it does not push the policy to stop there.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_ee = asset.data.body_link_lin_vel_w[:, _body_idx(asset_cfg), :]
    v_ref = env.command_manager.get_command("trajectory")[:, 3:6]
    t_hat = v_ref / (torch.linalg.norm(v_ref, dim=-1, keepdim=True) + eps)
    return (v_ee * t_hat).sum(dim=-1)
