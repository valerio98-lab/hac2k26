"""Kinematic helpers: linear-Jacobian manipulability and singularity penalty.

Both functions share a per-env scratch buffer for the warp Jacobian.
The buffer is lazy-allocated on first call and stored on the env so the
two terms (observation + reward) reuse the same compute path each step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _JacBuffers:
    """Lazy-allocated warp buffers + their torch views."""

    __slots__ = (
        "jacp_wp",
        "jacr_wp",
        "point_wp",
        "body_wp",
        "jacp_t",
        "jacr_t",
        "point_t",
        "body_id",
        "joint_dof_ids",
    )

    def __init__(
        self, env: "ManagerBasedRlEnv", body_id: int, joint_dof_ids: torch.Tensor
    ):
        nworld = env.num_envs
        nv = int(env.sim.mj_model.nv)
        with wp.ScopedDevice(env.sim.wp_device):
            self.jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self.jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
            self.point_wp = wp.zeros(nworld, dtype=wp.vec3)
            self.body_wp = wp.zeros(nworld, dtype=wp.int32)
            self.body_wp.fill_(body_id)
        self.jacp_t = wp.to_torch(self.jacp_wp)
        self.jacr_t = wp.to_torch(self.jacr_wp)
        self.point_t = wp.to_torch(self.point_wp).view(nworld, 3)
        self.body_id = body_id
        self.joint_dof_ids = joint_dof_ids


def _get_or_create_buffers(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    joint_pattern: str,
) -> _JacBuffers:
    """Per-env, per-body cached buffers, keyed off ``(body_id, joints)``."""
    cache = getattr(env, "_jac_buffers_cache", None)
    if cache is None:
        cache = {}
        env._jac_buffers_cache = cache

    asset = env.scene[asset_cfg.name]
    body_local_ids = asset_cfg.body_ids
    assert isinstance(body_local_ids, list) and len(body_local_ids) >= 1
    body_id = int(asset.indexing.body_ids[body_local_ids[0]].item())

    joint_local_ids, _ = asset.find_joints(joint_pattern)
    joint_local_ids = torch.tensor(joint_local_ids, device=env.device, dtype=torch.long)
    joint_dof_ids = asset.indexing.joint_v_adr[joint_local_ids]

    key = (body_id, joint_pattern)
    if key not in cache:
        cache[key] = _JacBuffers(env, body_id, joint_dof_ids)
    return cache[key]


def _compute_linear_jacobian(
    env: "ManagerBasedRlEnv",
    buf: _JacBuffers,
) -> torch.Tensor:
    """Returns J_pos with shape ``(N, 3, n_joints)``."""
    asset = env.scene["robot"]  # fixed for this task
    ee_pos = asset.data.body_link_pos_w[:, _local_body_id_from_buf(asset, buf), :]
    buf.point_t[:] = ee_pos
    with wp.ScopedDevice(env.sim.wp_device):
        mjwarp.jac(
            env.sim.wp_model,
            env.sim.wp_data,
            buf.jacp_wp,
            buf.jacr_wp,
            buf.point_wp,
            buf.body_wp,
        )
    return buf.jacp_t[:, :, buf.joint_dof_ids]


def _local_body_id_from_buf(asset, buf: _JacBuffers) -> int:
    """Map the global body id stored in the buffer back to the asset's
    local index (used by ``asset.data.body_link_pos_w``)."""
    global_ids = asset.indexing.body_ids
    # Linear search — only runs once per cache hit, asset has O(20) bodies.
    matches = (global_ids == buf.body_id).nonzero(as_tuple=False)
    return int(matches[0, 0].item())


def manipulability(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_pattern: str = "joint[1-7]",
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Linear-Jacobian manipulability 'w(q) = sqrt(det(J · J^T))'"""
    buf = _get_or_create_buffers(env, asset_cfg, joint_pattern)
    jacp = _compute_linear_jacobian(env, buf)  # (N, 3, 7)
    jjt = jacp @ jacp.transpose(-2, -1)  # (N, 3, 3)
    det = torch.linalg.det(jjt).clamp(min=0.0)
    return torch.sqrt(det + eps).unsqueeze(-1)


def singularity_penalty(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_pattern: str = "joint[1-7]",
    w_min: float = 0.04,
) -> torch.Tensor:

    w = manipulability(env, asset_cfg, joint_pattern).squeeze(-1)  # (N,)
    deficit = (w_min - w).clamp(min=0.0)
    return deficit * deficit
