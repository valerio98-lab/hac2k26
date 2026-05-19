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
    alpha_coarse: float = 30.0,
    alpha_fine: float = 400.0,
    fine_weight: float = 0.5,
) -> torch.Tensor:
    """Two-scale Gaussian kernel on EE position error.

    r = (1 - fw)*exp(-alpha_coarse*e^2) + fw*exp(-alpha_fine*e^2)

    The coarse term provides a usable gradient when the EE is far (>10 cm),
    while the fine term rewards sub-cm precision once close.
    """
    asset: Entity = env.scene[asset_cfg.name]
    p_ee = asset.data.body_link_pos_w[:, _body_idx(asset_cfg), :]
    p_ref = env.command_manager.get_command("trajectory")[:, 0:3]
    err_sq = torch.sum(torch.square(p_ee - p_ref), dim=-1)
    coarse = torch.exp(-alpha_coarse * err_sq)
    fine = torch.exp(-alpha_fine * err_sq)
    return (1.0 - fine_weight) * coarse + fine_weight * fine


def pos_error_l2_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L2 norm of EE position error (always-on dense signal). r = ||p_ee - p_ref||."""
    asset: Entity = env.scene[asset_cfg.name]
    p_ee = asset.data.body_link_pos_w[:, _body_idx(asset_cfg), :]
    p_ref = env.command_manager.get_command("trajectory")[:, 0:3]
    return torch.linalg.norm(p_ee - p_ref, dim=-1)


def orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    beta_coarse: float = 2.0,
    beta_fine: float = 20.0,
    fine_weight: float = 0.5,
) -> torch.Tensor:
    """Two-scale Gaussian kernel on the geodesic angle error."""
    asset: Entity = env.scene[asset_cfg.name]
    q_ee = asset.data.body_link_quat_w[:, _body_idx(asset_cfg), :]
    q_ref = env.command_manager.get_command("trajectory")[:, 7:11]
    theta_err = quat_error_magnitude(q_ee, q_ref)
    e2 = theta_err * theta_err
    coarse = torch.exp(-beta_coarse * e2)
    fine = torch.exp(-beta_fine * e2)
    return (1.0 - fine_weight) * coarse + fine_weight * fine


def action_jerk_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Discrete jerk on raw policy outputs: ||a_t - 2*a_{t-1} + a_{t-2}||^2.

    Penalises the second discrete derivative of the action sequence, which
    directly drives the high-frequency jitter visible at play time.
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    a_pp = env.action_manager.prev_prev_action
    return torch.sum(torch.square(a - 2.0 * a_prev + a_pp), dim=1)


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
