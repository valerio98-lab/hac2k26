"""Trajectory command for EE tracking.

Generates a piecewise reference signal for the end-effector through K random
waypoints in a configurable workspace box. Position uses rest-to-rest quintic
splines in cylindrical coordinates (r, θ, z) about the robot base z-axis. Orientation uses SLERP
between random quaternion waypoints with the same quintic smoothstep
time-reparametrization (so angular velocity is zero at every waypoint,
matching the position rest-to-rest behaviour).

Internally state is fully batched over envs:
    coeffs:      (num_envs, K-1, 6, 3) — quintic coefficients per segment
    q_start:     (num_envs, K-1, 4)    — start quaternion of each segment (wxyz)
    axis_angle:  (num_envs, K-1, 3)    — world-frame log(q_{i+1} q_i^{-1}),
                                         i.e. axis-angle of the relative rotation
    cum_times:   (K,)                  — shared since segment_duration is uniform

Resampling is auto-triggered by the command manager when 'time_left' hits
zero. 'resampling_time_range' is set to ``(total_duration, total_duration)``
so a new trajectory is sampled exactly when the previous one completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_box_minus,
    quat_box_plus,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class EETrackingCommand(CommandTerm):
    """Quintic-spline EE reference generator with phase, lookahead and SLERP
    orientation.

    Command tensor layout (column ranges):
        [0:3]                          p_ref      — target EE position (m)
        [3:6]                          v_ref      — target EE linear velocity (m/s)
        [6:7]                          phase      — normalised time in [0, 1]
        [7:11]                         q_ref      — target EE quaternion (wxyz)
        [11:14]                        omega_ref  — target angular velocity (rad/s, world)
        [14:14 + 3*lookahead_steps]    lookahead  — flat (L, 3) future positions
    """

    cfg: EETrackingCommandCfg

    def __init__(self, cfg: EETrackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.K = int(cfg.num_waypoints)
        if self.K < 2:
            raise ValueError(f"num_waypoints must be >= 2, got {self.K}")
        self.n_seg = self.K - 1
        self.T_seg = float(cfg.segment_duration)
        self.total_duration = self.n_seg * self.T_seg
        self.L = int(cfg.lookahead_steps)
        self.lookahead_dt = float(cfg.lookahead_dt)

        self._ws_low = torch.tensor(cfg.workspace_low, device=self.device)
        self._ws_high = torch.tensor(cfg.workspace_high, device=self.device)

        self._anchor = bool(cfg.anchor_first_waypoint)
        self._r_min = float(cfg.r_min)
        self._radius_start = float(cfg.radius_start)
        self._radius_end = float(cfg.radius_end)
        self._ori_radius_start = float(cfg.ori_radius_start)
        self._ori_radius_end = float(cfg.ori_radius_end)
        self._vel_scale_start = float(cfg.vel_scale_start)
        self._vel_scale_end = float(cfg.vel_scale_end)
        self._alpha_min_ratio = float(cfg.alpha_min_ratio)
        self._curriculum_steps = max(1, int(cfg.curriculum_steps))
        self._asset_name = cfg.asset_name
        self._body_name = cfg.body_name
        asset = env.scene[self._asset_name]
        self._body_id = asset.body_names.index(self._body_name)

        self._coeffs = torch.zeros(self.num_envs, self.n_seg, 6, 3, device=self.device)
        self._cum_times = torch.linspace(
            0.0, self.total_duration, self.K, device=self.device
        )

        self._q_start = torch.zeros(self.num_envs, self.n_seg, 4, device=self.device)
        self._q_start[..., 0] = 1.0  # identity init
        self._axis_angle = torch.zeros(self.num_envs, self.n_seg, 3, device=self.device)

        cmd_dim = 3 + 3 + 1 + 4 + 3 + 3 * self.L
        self._command = torch.zeros(self.num_envs, cmd_dim, device=self.device)

        self._anchor_dirty = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._in_reset = False

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | None = None):
        self._in_reset = True
        try:
            info = super().reset(env_ids)
        finally:
            self._in_reset = False
        return info

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return
        if self._in_reset:
            self._anchor_dirty[env_ids] = True
            return
        self._do_resample(env_ids)

    def _do_resample(self, env_ids: torch.Tensor) -> None:
        """Sample K waypoints (position + orientation) and precompute
        per-segment trajectory state for the given envs."""
        n = int(env_ids.numel())

        progress = min(
            1.0, float(self._env.common_step_counter) / self._curriculum_steps
        )
        radius = self._radius_start + progress * (self._radius_end - self._radius_start)
        ori_radius = self._ori_radius_start + progress * (
            self._ori_radius_end - self._ori_radius_start
        )
        vel_scale = self._vel_scale_start + progress * (
            self._vel_scale_end - self._vel_scale_start
        )

        asset = self._env.scene[self._asset_name]
        ee_pos = asset.data.body_link_pos_w[env_ids, self._body_id, :]  # (n, 3)
        ee_quat = asset.data.body_link_quat_w[env_ids, self._body_id, :]  # (n, 4)

        u = torch.rand(n, self.K, 3, device=self.device)
        wps_box = ee_pos.unsqueeze(1) + (2.0 * u - 1.0) * radius  # (n, K, 3)
        wps = torch.minimum(torch.maximum(wps_box, self._ws_low), self._ws_high)

        if self._anchor:
            wps = wps.clone()
            wps[:, 0, :] = ee_pos

        r = torch.linalg.norm(wps[..., :2], dim=-1)  # (n, K)
        r = torch.maximum(r, torch.full_like(r, self._r_min))
        theta = torch.atan2(wps[..., 1], wps[..., 0])  # (n, K) in (-π, π]

        dtheta = theta[:, 1:] - theta[:, :-1]
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
        theta_unwrap = torch.cat(
            [theta[:, :1], theta[:, :1] + torch.cumsum(dtheta, dim=1)], dim=1
        )
        wps_cyl = torch.stack([r, theta_unwrap, wps[..., 2]], dim=-1)  # (n, K, 3)

        # Centripetal Catmull-Rom velocities for interior waypoints
        T = self.T_seg
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        wp_vels = torch.zeros_like(wps_cyl)
        if self.K >= 3 and vel_scale > 0.0:
            r = self._alpha_min_ratio
            alpha = vel_scale * (
                r + (1.0 - r) * torch.rand(n, 1, 1, device=self.device)
            )
            wp_vels[:, 1:-1, :] = alpha * (
                (wps_cyl[:, 2:, :] - wps_cyl[:, :-2, :]) / (2.0 * T)
            )

        p_i = wps_cyl[:, :-1, :]
        p_ip1 = wps_cyl[:, 1:, :]
        v_i = wp_vels[:, :-1, :]
        v_ip1 = wp_vels[:, 1:, :]

        dp = p_ip1 - p_i - v_i * T
        dv = v_ip1 - v_i

        c0 = p_i
        c1 = v_i
        c2 = torch.zeros_like(p_i)
        c3 = 10.0 * dp / T3 - 4.0 * dv / T2
        c4 = -15.0 * dp / T4 + 7.0 * dv / T3
        c5 = 6.0 * dp / T5 - 3.0 * dv / T4

        coeffs = torch.stack([c0, c1, c2, c3, c4, c5], dim=2)  # (n, K-1, 6, 3) (r,θ,z)
        self._coeffs[env_ids] = coeffs

        rand_dir = torch.randn(n, self.K, 3, device=self.device)
        rand_dir = rand_dir / (rand_dir.norm(dim=-1, keepdim=True) + 1e-8)
        rand_mag = ori_radius * torch.rand(n, self.K, 1, device=self.device)
        deltas = (rand_dir * rand_mag).reshape(n * self.K, 3)
        ee_quat_rep = ee_quat.unsqueeze(1).expand(n, self.K, 4).reshape(n * self.K, 4)
        wps_q = quat_box_plus(ee_quat_rep, deltas).reshape(n, self.K, 4)
        if self._anchor:
            wps_q = wps_q.clone()
            wps_q[:, 0, :] = ee_quat

        q_i = wps_q[:, :-1, :].reshape(n * self.n_seg, 4)
        q_iplus1 = wps_q[:, 1:, :].reshape(n * self.n_seg, 4)
        axis_angle = quat_box_minus(q_iplus1, q_i).reshape(n, self.n_seg, 3)

        self._q_start[env_ids] = wps_q[:, :-1, :]
        self._axis_angle[env_ids] = axis_angle

    def _update_command(self) -> None:
        """Evaluate the reference signal at the current trajectory time."""
        if self._anchor_dirty.any():
            dirty_ids = self._anchor_dirty.nonzero(as_tuple=False).squeeze(-1)
            self._do_resample(dirty_ids)
            self._anchor_dirty[dirty_ids] = False

        t_now = (self.total_duration - self.time_left).clamp(0.0, self.total_duration)

        phase = (t_now / self.total_duration).unsqueeze(-1)  # (N, 1)

        p_ref, v_ref = self._eval_batched(t_now.unsqueeze(-1))  # (N, 1, 3)
        p_ref = p_ref.squeeze(1)
        v_ref = v_ref.squeeze(1)

        offsets = self.lookahead_dt * torch.arange(
            1, self.L + 1, device=self.device, dtype=t_now.dtype
        )  # (L,)
        t_look = t_now.unsqueeze(-1) + offsets.unsqueeze(0)  # (N, L)
        p_look, _ = self._eval_batched(t_look)
        lookahead = p_look.reshape(self.num_envs, -1)  # (N, L*3)

        seg_idx, local_t = self._segment_of(t_now)
        u = local_t / self.T_seg  # (N,)
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        s_dot = 30.0 * u * u * (1.0 - u) * (1.0 - u)

        env_idx = torch.arange(self.num_envs, device=self.device)
        q_start_seg = self._q_start[env_idx, seg_idx]  # (N, 4)
        aa_seg = self._axis_angle[env_idx, seg_idx]  # (N, 3)

        delta = s.unsqueeze(-1) * aa_seg  # (N, 3)
        q_ref = quat_box_plus(q_start_seg, delta)  # (N, 4)
        omega_ref = (s_dot.unsqueeze(-1) / self.T_seg) * aa_seg  # (N, 3) world

        self._command = torch.cat(
            [p_ref, v_ref, phase, q_ref, omega_ref, lookahead], dim=-1
        )

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer) -> None:

        for env_idx in visualizer.get_env_indices(self.num_envs):
            N_samples = 300
            t = torch.linspace(
                0.0, self.total_duration, N_samples, device=self.device
            ).unsqueeze(
                0
            )  # (1, N)
            # Temporarily evaluate just this env.
            coeffs = self._coeffs[env_idx : env_idx + 1]
            cum = self._cum_times
            interior = cum[1:-1]
            seg = torch.searchsorted(
                interior, t.clamp(0.0, self.total_duration), right=True
            )
            seg = seg.clamp(0, self.n_seg - 1)
            local_t = t - cum[seg]
            powers = torch.stack([local_t**i for i in range(6)], dim=-1)  # (1, N, 6)
            selected = coeffs[0, seg[0]]  # (N, 6, 3) in (r, θ, z)
            pts_cyl = torch.einsum("ni,nij->nj", powers[0], selected)  # (N, 3)
            pts = (
                torch.stack(
                    [
                        pts_cyl[:, 0] * torch.cos(pts_cyl[:, 1]),
                        pts_cyl[:, 0] * torch.sin(pts_cyl[:, 1]),
                        pts_cyl[:, 2],
                    ],
                    dim=-1,
                )
                .cpu()
                .numpy()
            )

            for p in pts:
                visualizer.add_sphere(
                    center=p, radius=0.008, color=(0.2, 0.9, 0.3, 0.8)
                )

            p_ref = self._command[env_idx, 0:3].cpu().numpy()
            visualizer.add_sphere(
                center=p_ref, radius=0.025, color=(0.9, 0.2, 0.2, 1.0), label="p_ref"
            )

    def _segment_of(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a per-env time t (N,) to (segment_index, local_time)."""
        t_clamped = t.clamp(0.0, self.total_duration)
        interior = self._cum_times[1:-1]
        seg_idx = torch.searchsorted(interior, t_clamped, right=True)
        seg_idx = seg_idx.clamp(0, self.n_seg - 1)
        local_t = t_clamped - self._cum_times[seg_idx]
        return seg_idx, local_t

    def _eval_batched(
        self, t_per_env: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the quintic spline at multiple times per env.
        The spline lives in cylindrical (r, θ, z) coords this method
        converts back to cartesian (x, y, z)

        Args:
            t_per_env: shape (N, L). Times to evaluate per env.

        Returns:
            (pos, vel) each of shape (N, L, 3), in world-frame xyz.
        """
        N, L = t_per_env.shape
        t_clamped = t_per_env.clamp(0.0, self.total_duration)

        interior = self._cum_times[1:-1]  # (K-2,)
        seg_idx = torch.searchsorted(interior, t_clamped, right=True)
        seg_idx = seg_idx.clamp(0, self.n_seg - 1)

        local_t = t_clamped - self._cum_times[seg_idx]  # (N, L)

        env_grid = torch.arange(N, device=self.device).unsqueeze(-1).expand(N, L)
        selected = self._coeffs[env_grid, seg_idx]  # (N, L, 6, 3) in (r,θ,z)

        powers = torch.stack([local_t**i for i in range(6)], dim=-1)  # (N, L, 6)
        pos_cyl = torch.einsum("nli,nlij->nlj", powers, selected)

        vel_coeffs = torch.stack(
            [(i + 1) * selected[..., i + 1, :] for i in range(5)], dim=-2
        )  # (N, L, 5, 3)
        vel_cyl = torch.einsum("nli,nlij->nlj", powers[..., :5], vel_coeffs)

        # Cylindrical → cartesian:
        #   x = r cos θ        x_dot = r_dot cos θ − r θ_dot sin θ
        #   y = r sin θ        y_dot = r_dot sin θ + r θ_dot cos θ
        #   z = z              z_dot = z_dot
        r = pos_cyl[..., 0]
        th = pos_cyl[..., 1]
        z = pos_cyl[..., 2]
        rd = vel_cyl[..., 0]
        thd = vel_cyl[..., 1]
        zd = vel_cyl[..., 2]
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)
        pos = torch.stack([r * cos_th, r * sin_th, z], dim=-1)
        vel = torch.stack(
            [rd * cos_th - r * thd * sin_th, rd * sin_th + r * thd * cos_th, zd],
            dim=-1,
        )
        return pos, vel


@dataclass(kw_only=True)
class EETrackingCommandCfg(CommandTermCfg):
    """Configuration for `EETrackingCommand`."""

    num_waypoints: int = 5
    """Number of random waypoints per trajectory (K). K-1 segments."""

    segment_duration: float = 1.5
    """Duration of each segment in seconds. Total trajectory: (K-1) * segment_duration."""

    workspace_low: tuple[float, float, float] = (0.30, -0.30, 0.20)
    """Lower corner of the random-waypoint sampling box (m, world frame)."""

    workspace_high: tuple[float, float, float] = (0.60, 0.30, 0.60)
    """Upper corner of the random-waypoint sampling box (m, world frame)."""

    lookahead_steps: int = 5
    """Number of future reference samples exposed to the policy."""

    lookahead_dt: float = 0.05
    """Time delta between consecutive lookahead samples (s)."""

    anchor_first_waypoint: bool = True
    """If True, the first waypoint of every sampled trajectory is the current
    EE pose (pos + quat), so the trajectory starts with zero initial error."""

    r_min: float = 0.20
    """Minimum cylindrical radius (m) of waypoints from the robot base z-axis.
    Waypoints sampled with r < r_min are projected radially outward to r_min."""

    radius_start: float = 0.05
    """Initial half-edge (m) of the cube around the current EE position from
    which subsequent waypoints are sampled. Trajectories then get clamped to
    [workspace_low, workspace_high]."""

    radius_end: float = 0.30
    """Final half-edge (m) of the sampling cube after curriculum ramp-up."""

    ori_radius_start: float = 0.15
    """Initial max angular perturbation (rad) of waypoint quaternions around
    the current EE orientation."""

    ori_radius_end: float = 0.80
    """Final max angular perturbation (rad) after curriculum ramp-up
    (~45 deg). Kept moderate so targets stay reachable for the arm."""

    vel_scale_start: float = 0.0
    """Initial through-velocity scale. v_i at an interior waypoint is
    ``alpha * (p_{i+1} - p_{i-1}) / (2 T_seg)`` with alpha sampled in
    [0, vel_scale]. ``vel_scale=0`` reduces to pure rest-to-rest"""

    vel_scale_end: float = 0.0
    """Final through-velocity scale."""

    alpha_min_ratio: float = 0.0

    curriculum_steps: int = 2_000_000
    """Number of env-steps over which radius linearly ramps from start to end."""

    asset_name: str = "robot"
    body_name: str = "hand"

    resampling_time_range: tuple[float, float] = field(default=(0.0, 0.0))
    """Auto-set in __post_init__ to match trajectory total duration."""

    def __post_init__(self) -> None:
        total = (self.num_waypoints - 1) * self.segment_duration
        if self.resampling_time_range == (0.0, 0.0):
            self.resampling_time_range = (total, total)

    def build(self, env: ManagerBasedRlEnv) -> EETrackingCommand:
        return EETrackingCommand(self, env)
