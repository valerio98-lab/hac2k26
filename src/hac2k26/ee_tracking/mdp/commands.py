"""Trajectory command for EE tracking.

Supports two waypoint-sampling regimes selected via ``sampling_mode``:

- ``"local"`` — **Cartesian** quintic interpolation between waypoints sampled
  in a ``±radius`` cube around the CURRENT EE position, clamped to a
  workspace box ``[workspace_low, workspace_high]``. Produces local, smooth
  motions; well-suited to small circles and figure-8s near the EE.

- ``"global"`` — **Cylindrical** ``(r, θ, z)`` quintic interpolation between
  waypoints sampled globally with ``r ∈ [r_min, r_max]``, ``θ`` accumulated
  from the current EE angle via bounded Δθ increments, ``z ∈ [z_min, z_max]``.
  Cartesian ``(x, y, z)`` reconstruction at eval time via
  ``(r·cosθ, r·sinθ, z)`` guarantees paths with ``r ≥ r_min`` never cut
  through the base. Produces large angular sweeps; well-suited to circles
  centered on the base axis.

Orientation is identical in both modes: SLERP between random quaternion
waypoints with a quintic smoothstep time-reparametrization (so angular
velocity is zero at every waypoint, matching the position rest-to-rest
behaviour).

Internally state is fully batched over envs:
    coeffs:      (num_envs, K-1, 6, 3) — quintic coefficients per segment,
                                         channels (x, y, z) in ``local`` mode
                                         or (r, θ, z) in ``global`` mode
    q_start:     (num_envs, K-1, 4)    — start quaternion of each segment (wxyz)
    axis_angle:  (num_envs, K-1, 3)    — world-frame log(q_{i+1} q_i^{-1})
    cum_times:   (K,)                  — shared since segment_duration is uniform

Resampling is auto-triggered by the command manager when ``time_left`` hits
zero. ``resampling_time_range`` is set to ``(total_duration, total_duration)``
so a new trajectory is sampled exactly when the previous one completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_box_minus,
    quat_box_plus,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class EETrackingCommand(CommandTerm):
    """Quintic-spline EE reference generator with phase, lookahead and SLERP
    orientation. Supports ``local`` (cartesian) and ``global`` (cylindrical)
    waypoint-sampling modes.

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

        if cfg.sampling_mode not in ("local", "global"):
            raise ValueError(
                f"sampling_mode must be 'local' or 'global', got {cfg.sampling_mode!r}"
            )
        self._mode = cfg.sampling_mode

        self.K = int(cfg.num_waypoints)
        if self.K < 2:
            raise ValueError(f"num_waypoints must be >= 2, got {self.K}")
        self.n_seg = self.K - 1
        self.T_seg = float(cfg.segment_duration)
        self.total_duration = self.n_seg * self.T_seg
        self.L = int(cfg.lookahead_steps)
        self.lookahead_dt = float(cfg.lookahead_dt)

        self._anchor = bool(cfg.anchor_first_waypoint)

        # Local-mode params.
        self._ws_low = torch.tensor(cfg.workspace_low, device=self.device)
        self._ws_high = torch.tensor(cfg.workspace_high, device=self.device)
        self._radius_start = float(cfg.radius_start)
        self._radius_end = float(cfg.radius_end)

        # Global-mode params.
        self._r_min = float(cfg.r_min)
        self._r_max = float(cfg.r_max)
        self._z_min = float(cfg.z_min)
        self._z_max = float(cfg.z_max)
        self._dtheta_max_start = float(cfg.dtheta_max_start)
        self._dtheta_max_end = float(cfg.dtheta_max_end)
        # Absolute clamp on the cumulative (unwrapped) theta. The EE polar
        # angle around the base z-axis must be reachable by joint1, whose
        # soft limit on the Panda is ±0.9 · 2.8973 ≈ ±2.608 rad. Trajectories
        # whose unwrapped theta would exceed this are geometrically
        # infeasible no matter how the redundancy is resolved.
        self._theta_max = float(cfg.theta_max_rad)
        if self._theta_max <= 0.0:
            raise ValueError(
                f"theta_max_rad must be > 0, got {self._theta_max}"
            )

        # Shared.
        self._ori_radius_start = float(cfg.ori_radius_start)
        self._ori_radius_end = float(cfg.ori_radius_end)
        self._vel_scale_start = float(cfg.vel_scale_start)
        self._vel_scale_end = float(cfg.vel_scale_end)
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

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _curriculum_progress(self) -> float:
        return min(1.0, float(self._env.common_step_counter) / self._curriculum_steps)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return
        if self._mode == "local":
            self._resample_local_cartesian(env_ids, n)
        else:
            self._resample_global_cylindrical(env_ids, n)

    def _resample_local_cartesian(self, env_ids: torch.Tensor, n: int) -> None:
        """Sample K waypoints in a ±radius cube around the current EE,
        clamped to ``[workspace_low, workspace_high]``. Quintic interpolation
        is done in cartesian (x, y, z) channels."""
        progress = self._curriculum_progress()
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

        self._fit_quintic_and_store(env_ids, wps, vel_scale, n)
        self._sample_orientation_and_store(env_ids, ee_quat, ori_radius, n)

    def _resample_global_cylindrical(self, env_ids: torch.Tensor, n: int) -> None:
        """Sample K cylindrical waypoints. θ is kept UNWRAPPED (cumulative sum of
        bounded Δθ increments from the EE angle), so the per-segment quintic
        interpolates the short angular path between two waypoints — never
        going the long way around via the modulo-2π gap."""
        progress = self._curriculum_progress()
        dtheta_max = self._dtheta_max_start + progress * (
            self._dtheta_max_end - self._dtheta_max_start
        )
        ori_radius = self._ori_radius_start + progress * (
            self._ori_radius_end - self._ori_radius_start
        )
        vel_scale = self._vel_scale_start + progress * (
            self._vel_scale_end - self._vel_scale_start
        )

        asset = self._env.scene[self._asset_name]
        ee_pos = asset.data.body_link_pos_w[env_ids, self._body_id, :]  # (n, 3)
        ee_quat = asset.data.body_link_quat_w[env_ids, self._body_id, :]  # (n, 4)

        curr_theta = torch.atan2(ee_pos[:, 1], ee_pos[:, 0]).unsqueeze(-1)  # (n, 1)

        # 50/50 mix of per-trajectory sampling regimes:
        #  - "biased": one shared sign per trajectory and per-segment magnitudes
        #    in [0, dtheta_max] — produces consistently CCW or CW rotation
        #    (50/50 between the two), which is the regime needed for clean
        #    circular eval trajectories.
        #  - "wobble": per-segment Δθ in [-dtheta_max, +dtheta_max] — a random
        #    walk in θ that produces direction reversals, useful for figure-8
        #    and moving-target style references.
        sign = torch.where(
            torch.rand(n, 1, device=self.device) < 0.5,
            -torch.ones(n, 1, device=self.device),
            torch.ones(n, 1, device=self.device),
        )
        biased = sign * dtheta_max * torch.rand(n, self.K - 1, device=self.device)
        wobble = (
            2.0 * torch.rand(n, self.K - 1, device=self.device) - 1.0
        ) * dtheta_max
        use_biased = torch.rand(n, 1, device=self.device) < 0.5
        delta_theta = torch.where(use_biased, biased, wobble)

        # Cumulative unwrapped theta. The first column equals curr_theta
        # by construction (anchor). Subsequent columns are clamped to
        # ±theta_max so the trajectory stays inside joint1's soft limit;
        # if the cumulative sweep saturates, the spline still moves in r
        # and z but stops rotating around the base — that is the correct
        # graceful-degradation behaviour at the joint limit. The anchor
        # column is left untouched even if curr_theta would otherwise be
        # out of range (rare; preserves the start-at-EE invariant).
        cumsum_delta = torch.cumsum(delta_theta, dim=-1)  # (n, K-1)
        theta_tail = (curr_theta + cumsum_delta).clamp(
            -self._theta_max, self._theta_max
        )  # (n, K-1)
        theta_i = torch.cat([curr_theta, theta_tail], dim=-1)  # (n, K)
        r_i = self._r_min + (self._r_max - self._r_min) * torch.rand(
            n, self.K, device=self.device
        )
        z_i = self._z_min + (self._z_max - self._z_min) * torch.rand(
            n, self.K, device=self.device
        )

        if self._anchor:
            r_curr = ee_pos[:, :2].norm(dim=-1)
            r_i = r_i.clone()
            z_i = z_i.clone()
            r_i[:, 0] = r_curr
            z_i[:, 0] = ee_pos[:, 2]
            # theta_i[:, 0] is already curr_theta by construction.

        wps_cyl = torch.stack([r_i, theta_i, z_i], dim=-1)  # (n, K, 3)

        self._fit_quintic_and_store(env_ids, wps_cyl, vel_scale, n)
        self._sample_orientation_and_store(env_ids, ee_quat, ori_radius, n)

    def _fit_quintic_and_store(
        self,
        env_ids: torch.Tensor,
        wps: torch.Tensor,
        vel_scale: float,
        n: int,
    ) -> None:
        """Fit quintic Hermite per-segment in whatever channel space ``wps``
        lives in (cartesian for local mode, cylindrical for global mode).
        Centripetal Catmull-Rom velocities at INTERIOR waypoints; v=0 at the
        two endpoints so trajectories start and end at rest."""
        T = self.T_seg
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        wp_vels = torch.zeros_like(wps)  # (n, K, 3), endpoints stay 0
        if self.K >= 3 and vel_scale > 0.0:
            alpha = vel_scale * torch.rand(n, 1, 1, device=self.device)
            wp_vels[:, 1:-1, :] = alpha * ((wps[:, 2:, :] - wps[:, :-2, :]) / (2.0 * T))

        p_i = wps[:, :-1, :]
        p_ip1 = wps[:, 1:, :]
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

        coeffs = torch.stack([c0, c1, c2, c3, c4, c5], dim=2)  # (n, K-1, 6, 3)
        self._coeffs[env_ids] = coeffs

    def _sample_orientation_and_store(
        self,
        env_ids: torch.Tensor,
        ee_quat: torch.Tensor,
        ori_radius: float,
        n: int,
    ) -> None:
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

        seg_idx, local_t = self._segment_of(t_now)  # (N,), (N,)
        u = local_t / self.T_seg  # (N,)
        s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)  # (N,)
        s_dot = 30.0 * u * u * (1.0 - u) * (1.0 - u)  # (N,)

        env_idx = torch.arange(self.num_envs, device=self.device)
        q_start_seg = self._q_start[env_idx, seg_idx]  # (N, 4)
        aa_seg = self._axis_angle[env_idx, seg_idx]  # (N, 3)

        delta = s.unsqueeze(-1) * aa_seg  # (N, 3) — world-frame increment
        q_ref = quat_box_plus(q_start_seg, delta)  # (N, 4)
        omega_ref = (s_dot.unsqueeze(-1) / self.T_seg) * aa_seg  # (N, 3) world

        self._command = torch.cat(
            [p_ref, v_ref, phase, q_ref, omega_ref, lookahead], dim=-1
        )

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer) -> None:
        N_samples = 300
        t = torch.linspace(
            0.0, self.total_duration, N_samples, device=self.device
        )  # (N,)

        t_batched = t.unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, N)
        pts_all, _ = self._eval_batched(t_batched)  # (num_envs, N, 3) cartesian

        for env_idx in visualizer.get_env_indices(self.num_envs):
            pts = pts_all[env_idx].cpu().numpy()
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
        """Evaluate the quintic at multiple times per env and return the
        **cartesian** position and velocity. In ``global`` mode the polynomial
        channels are (r, θ, z) and are converted to cartesian via cos/sin
        plus the velocity chain rule.

        Args:
            t_per_env: shape (N, L). Times to evaluate per env.

        Returns:
            (pos, vel) each of shape (N, L, 3) in world cartesian frame.
        """
        N, L = t_per_env.shape
        t_clamped = t_per_env.clamp(0.0, self.total_duration)

        interior = self._cum_times[1:-1]  # (K-2,)
        seg_idx = torch.searchsorted(interior, t_clamped, right=True)
        seg_idx = seg_idx.clamp(0, self.n_seg - 1)

        local_t = t_clamped - self._cum_times[seg_idx]  # (N, L)

        env_grid = torch.arange(N, device=self.device).unsqueeze(-1).expand(N, L)
        selected = self._coeffs[env_grid, seg_idx]  # (N, L, 6, 3)

        powers = torch.stack([local_t**i for i in range(6)], dim=-1)  # (N, L, 6)
        channels = torch.einsum("nli,nlij->nlj", powers, selected)  # (N, L, 3)

        vel_coeffs = torch.stack(
            [(i + 1) * selected[..., i + 1, :] for i in range(5)], dim=-2
        )  # (N, L, 5, 3)
        channels_vel = torch.einsum("nli,nlij->nlj", powers[..., :5], vel_coeffs)

        if self._mode == "local":
            # Channels are already (x, y, z).
            return channels, channels_vel

        # Global mode: channels are (r, θ, z). Convert to cartesian.
        r = channels[..., 0]
        th = channels[..., 1]
        z = channels[..., 2]
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)
        pos = torch.stack([r * cos_th, r * sin_th, z], dim=-1)

        # Chain-rule for velocity:
        #   dx/dt = dr/dt cosθ - r sinθ dθ/dt
        #   dy/dt = dr/dt sinθ + r cosθ dθ/dt
        #   dz/dt = dz/dt
        dr = channels_vel[..., 0]
        dth = channels_vel[..., 1]
        dz = channels_vel[..., 2]
        vx = dr * cos_th - r * sin_th * dth
        vy = dr * sin_th + r * cos_th * dth
        vel = torch.stack([vx, vy, dz], dim=-1)

        return pos, vel


@dataclass(kw_only=True)
class EETrackingCommandCfg(CommandTermCfg):
    """Configuration for `EETrackingCommand`."""

    sampling_mode: Literal["local", "global"] = "global"
    """Waypoint sampling regime.

    - ``"local"``  — cartesian quintic, waypoints in a ``±radius`` cube around
                     the current EE, clamped to ``[workspace_low, workspace_high]``.
    - ``"global"`` — cylindrical quintic, waypoints sampled globally with
                     ``r ∈ [r_min, r_max]``, θ accumulated by bounded Δθ from
                     the current EE angle, ``z ∈ [z_min, z_max]``."""

    num_waypoints: int = 5
    """Number of random waypoints per trajectory (K). K-1 segments."""

    segment_duration: float = 1.5
    """Duration of each segment in seconds. Total trajectory: (K-1) * segment_duration."""

    workspace_low: tuple[float, float, float] = (0.20, -0.40, 0.10)
    """Lower corner of the cartesian clamping box. **Local mode only.**"""

    workspace_high: tuple[float, float, float] = (0.70, 0.40, 0.80)
    """Upper corner of the cartesian clamping box. **Local mode only.**"""

    lookahead_steps: int = 5
    """Number of future reference samples exposed to the policy."""

    lookahead_dt: float = 0.05
    """Time delta between consecutive lookahead samples (s)."""

    anchor_first_waypoint: bool = True
    """If True, the first waypoint of every sampled trajectory is the current
    EE pose (pos + quat), so the trajectory starts with zero initial error."""

    # ----- Local-mode params ---------------------------------------------------

    radius_start: float = 0.15
    """Initial half-edge (m) of the cube around the current EE position from
    which subsequent waypoints are sampled. **Local mode only.**"""

    radius_end: float = 0.45
    """Final half-edge (m) of the sampling cube after curriculum ramp-up.
    **Local mode only.**"""

    # ----- Global-mode params --------------------------------------------------

    r_min: float = 0.35
    """Inner radial bound (m, world XY plane from base) for waypoint sampling.
    **Global mode only.**"""

    r_max: float = 0.75
    """Outer radial bound (m). Should respect the arm's reachable XY radius.
    **Global mode only.**"""

    z_min: float = 0.10
    """Lower z bound (m). **Global mode only.**"""

    z_max: float = 0.90
    """Upper z bound (m). **Global mode only.**"""

    dtheta_max_start: float = 0.5
    """Initial maximum per-segment angular increment Δθ_max (rad).
    **Global mode only.**"""

    dtheta_max_end: float = 2.0
    """Final Δθ_max (rad). **Global mode only.**"""

    theta_max_rad: float = 2.5
    """Absolute clamp on the cumulative (unwrapped) waypoint theta around
    the base z-axis. Default 2.5 rad sits inside the Panda joint1 soft
    limit of 0.9 · 2.8973 ≈ 2.608 rad. Trajectories whose unwrapped theta
    would exceed this are not reachable by joint1, so the cumulative sum
    is clamped before spline fitting. **Global mode only.**"""

    # ----- Shared --------------------------------------------------------------

    ori_radius_start: float = 0.15
    """Initial max angular perturbation (rad) of waypoint quaternions around
    the current EE orientation."""

    ori_radius_end: float = 0.80
    """Final max angular perturbation (rad) after curriculum ramp-up."""

    vel_scale_start: float = 0.0
    """Initial through-velocity scale. v_i at an interior waypoint is
    ``alpha * (p_{i+1} - p_{i-1}) / (2 T_seg)`` with alpha sampled in
    [0, vel_scale]."""

    vel_scale_end: float = 0.0
    """Final through-velocity scale."""

    curriculum_steps: int = 2_000_000
    """Number of env-steps over which curriculum params linearly ramp."""

    asset_name: str = "robot"
    """Scene entity to query for the EE pose at resample time."""

    body_name: str = "hand"
    """Body of ``asset_name`` whose world-frame pose anchors the trajectory."""

    resampling_time_range: tuple[float, float] = field(default=(0.0, 0.0))
    """Auto-set in __post_init__ to match trajectory total duration."""

    def __post_init__(self) -> None:
        total = (self.num_waypoints - 1) * self.segment_duration
        if self.resampling_time_range == (0.0, 0.0):
            self.resampling_time_range = (total, total)

    def build(self, env: ManagerBasedRlEnv) -> EETrackingCommand:
        return EETrackingCommand(self, env)
