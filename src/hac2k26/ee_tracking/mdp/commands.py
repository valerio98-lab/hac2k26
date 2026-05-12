"""Trajectory command for EE tracking.

The command samples a piecewise quintic spline through K random waypoints in a
configurable workspace box and exposes the reference signal — current target
position, velocity, phase variable phi in [0, 1], and a lookahead window of
future positions — as a single tensor read by observations and rewards.

Internally state is fully batched over envs:
    coeffs: (num_envs, K-1, 6, 3) — quintic coefficients per env, per segment
    cum_times: (K,) — shared since segment_duration is uniform

Resampling is triggered automatically by the command manager when ``time_left``
hits zero. We set ``resampling_time_range`` to (total_duration, total_duration)
so the timer matches the trajectory length and a new spline is built whenever
the previous one completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class EETrackingCommand(CommandTerm):
    """Quintic-spline EE reference generator with phase + lookahead window.

    Command tensor layout (column ranges):
        [0:3]                       p_ref  — target EE position (m)
        [3:6]                       v_ref  — target EE velocity (m/s)
        [6:7]                       phase  — normalised time in [0, 1]
        [7:7 + 3*lookahead_steps]   lookahead — flattened (L, 3) future positions
                                                sampled at t + i*lookahead_dt
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

        # Workspace box.
        self._ws_low = torch.tensor(cfg.workspace_low, device=self.device)
        self._ws_high = torch.tensor(cfg.workspace_high, device=self.device)

        # Per-env trajectory state.
        self._coeffs = torch.zeros(self.num_envs, self.n_seg, 6, 3, device=self.device)
        # Cumulative segment times are uniform across envs (shared (K,) buffer).
        self._cum_times = torch.linspace(
            0.0, self.total_duration, self.K, device=self.device
        )

        # Cached current reference (filled in _update_command).
        self._p_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._v_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._lookahead = torch.zeros(self.num_envs, self.L * 3, device=self.device)

        cmd_dim = (
            3 + 3 + 1 + 3 * self.L
        )  # p_ref(3) + v_ref(3) + phase(1) + lookahead(L*3)
        self._command = torch.zeros(self.num_envs, cmd_dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample K random waypoints in the workspace and rebuild quintic
        coefficients for the given envs (rest-to-rest BCs)."""
        n = int(env_ids.numel())
        if n == 0:
            return

        # Random waypoints in workspace box.
        u = torch.rand(n, self.K, 3, device=self.device)
        wps = self._ws_low + u * (self._ws_high - self._ws_low)  # (n, K, 3)

        # Closed-form rest-to-rest coefficients (v0=v1=0, a0=a1=0).
        T = self.T_seg
        T3 = T * T * T
        T4 = T3 * T
        T5 = T4 * T
        dp = wps[:, 1:, :] - wps[:, :-1, :]  # (n, K-1, 3)
        zeros_seg = torch.zeros_like(dp)
        c0 = wps[:, :-1, :]
        c1 = zeros_seg
        c2 = zeros_seg
        c3 = 10.0 * dp / T3
        c4 = -15.0 * dp / T4
        c5 = 6.0 * dp / T5
        coeffs = torch.stack([c0, c1, c2, c3, c4, c5], dim=2)  # (n, K-1, 6, 3)

        self._coeffs[env_ids] = coeffs

    def _update_command(self) -> None:
        """Evaluate spline at current trajectory time + lookahead window."""
        # Current time within trajectory: total_duration - time_left.
        t_now = (self.total_duration - self.time_left).clamp(0.0, self.total_duration)

        # Phase phi in [0, 1].
        phase = (t_now / self.total_duration).unsqueeze(-1)  # (N, 1)

        # Current ref pos/vel.
        p_ref, v_ref = self._eval_batched(t_now.unsqueeze(-1))  # (N, 1, 3)
        p_ref = p_ref.squeeze(1)
        v_ref = v_ref.squeeze(1)

        # Lookahead window: t_now + dt, t_now + 2 dt, ..., t_now + L dt.
        offsets = self.lookahead_dt * torch.arange(
            1, self.L + 1, device=self.device, dtype=t_now.dtype
        )  # (L,)
        t_look = t_now.unsqueeze(-1) + offsets.unsqueeze(0)  # (N, L)
        p_look, _ = self._eval_batched(t_look)  # (N, L, 3)
        lookahead = p_look.reshape(self.num_envs, -1)  # (N, L*3)

        self._p_ref = p_ref
        self._v_ref = v_ref
        self._phase = phase
        self._lookahead = lookahead
        self._command = torch.cat([p_ref, v_ref, phase, lookahead], dim=-1)

    def _update_metrics(self) -> None:
        pass

    def _eval_batched(
        self, t_per_env: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the per-env splines at multiple time points.

        Args:
            t_per_env: shape (N, L). Times to evaluate per env. Will be clamped
                to [0, total_duration].

        Returns:
            (pos, vel) each of shape (N, L, 3).
        """
        N, L = t_per_env.shape
        t_clamped = t_per_env.clamp(0.0, self.total_duration)

        # Find segment per (env, sample). Search interior boundaries.
        interior = self._cum_times[1:-1]  # (K-2,)
        seg_idx = torch.searchsorted(interior, t_clamped, right=True)  # (N, L)
        seg_idx = seg_idx.clamp(0, self.n_seg - 1)

        # Local time within segment.
        local_t = t_clamped - self._cum_times[seg_idx]  # (N, L)

        # Gather coefficients per (env, sample): (N, L, 6, 3).
        env_grid = (
            torch.arange(N, device=self.device).unsqueeze(-1).expand(N, L)
        )  # (N, L)
        selected = self._coeffs[env_grid, seg_idx]  # (N, L, 6, 3)

        # Power matrix and polynomial evaluation.
        powers = torch.stack([local_t**i for i in range(6)], dim=-1)  # (N, L, 6)
        pos = torch.einsum("nli,nlij->nlj", powers, selected)  # (N, L, 3)

        vel_coeffs = torch.stack(
            [(i + 1) * selected[..., i + 1, :] for i in range(5)], dim=-2
        )  # (N, L, 5, 3)
        vel = torch.einsum("nli,nlij->nlj", powers[..., :5], vel_coeffs)

        return pos, vel


@dataclass(kw_only=True)
class EETrackingCommandCfg(CommandTermCfg):
    """Configuration for `EETrackingCommand`."""

    num_waypoints: int = 5
    """Number of random waypoints per trajectory (K). K-1 quintic segments."""

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

    resampling_time_range: tuple[float, float] = field(default=(0.0, 0.0))
    """Auto-set in __post_init__ to match trajectory total duration."""

    def __post_init__(self) -> None:
        total = (self.num_waypoints - 1) * self.segment_duration
        if self.resampling_time_range == (0.0, 0.0):
            self.resampling_time_range = (total, total)

    def build(self, env: ManagerBasedRlEnv) -> EETrackingCommand:
        return EETrackingCommand(self, env)
