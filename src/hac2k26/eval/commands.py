from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from hac2k26.eval.trajectories import (
    TrajectoryAnchor,
    _BaseDeterministicTraj,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


TrajectoryFactory = Callable[[], _BaseDeterministicTraj]


class DeterministicEETrackingCommand(CommandTerm):
    """Plays back a single analytic trajectory anchored to the EE pose."""

    cfg: "DeterministicEETrackingCommandCfg"

    def __init__(
        self,
        cfg: "DeterministicEETrackingCommandCfg",
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(cfg, env)

        if cfg.trajectory_factory is None:
            raise ValueError(
                "DeterministicEETrackingCommandCfg.trajectory_factory must be set "
                "before building the command (call cfg.set_trajectory(...))."
            )
        self._traj = cfg.trajectory_factory()
        self.L = int(cfg.lookahead_steps)
        self.lookahead_dt = float(cfg.lookahead_dt)

        asset = env.scene[cfg.asset_name]
        self._body_id = asset.body_names.index(cfg.body_name)
        self._asset_name = cfg.asset_name

        self._step_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        cmd_dim = 3 + 3 + 1 + 4 + 3 + 3 * self.L
        self._command = torch.zeros(self.num_envs, cmd_dim, device=self.device)

        self._ghost_n = int(cfg.ghost_samples)
        self._ghost_points: torch.Tensor | None = None  # (num_envs, ghost_n, 3)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._step_count[env_ids] = 0

    def _update_command(self) -> None:
        # Lazy re-anchor: any env at t=0 reads the (now-current) EE pose.
        if (self._step_count == 0).any():
            asset = self._env.scene[self._asset_name]
            ee_pos = asset.data.body_link_pos_w[:, self._body_id, :]
            ee_quat = asset.data.body_link_quat_w[:, self._body_id, :]
            print(
                f"Anchoring trajectory at EE pos {ee_pos.cpu().numpy()}",
                flush=True,
            )
            self._traj.set_anchor(TrajectoryAnchor(pos=ee_pos, quat=ee_quat))

            # Pre-sample the full path for the ghost overlay (single-env eval).
            t_grid = torch.linspace(
                0.0, self._traj.T_total, self._ghost_n, device=self.device
            )
            p_grid, _, _, _ = self._traj.evaluate(t_grid)
            self._ghost_points = p_grid.detach().clone()

        step_dt = self._env.step_dt
        t = self._step_count.to(dtype=torch.float32) * step_dt  # (N,)

        p, v, q, omega = self._traj.evaluate(t)
        phase = (t / self._traj.T_total).clamp(0.0, 1.0).unsqueeze(-1)

        # Lookahead: future positions at t + k * lookahead_dt.
        offsets = self.lookahead_dt * torch.arange(
            1, self.L + 1, device=self.device, dtype=t.dtype
        )
        t_look = (t.unsqueeze(-1) + offsets.unsqueeze(0)).reshape(-1)  # (N*L,)
        p_look, _, _, _ = self._traj.evaluate(t_look)
        lookahead = p_look.reshape(self.num_envs, self.L * 3)

        self._command = torch.cat([p, v, phase, q, omega, lookahead], dim=-1)
        self._step_count += 1

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        if self._ghost_points is None:
            return
        pts = self._ghost_points.detach().cpu().numpy()

        path_color = (0.10, 0.55, 0.95, 0.60)  # cyan-ish, semi-transparent
        seg_radius = 0.004
        stride = 1 if self._ghost_n <= 160 else 2
        for i in range(0, pts.shape[0] - 1, stride):
            visualizer.add_cylinder(
                start=pts[i],
                end=pts[i + 1],
                radius=seg_radius,
                color=path_color,
            )

        visualizer.add_sphere(
            center=pts[0],
            radius=0.012,
            color=(0.30, 0.30, 0.30, 1.0),
            label="start",
        )

        cmd = self._command[0].detach().cpu().numpy()
        p_ref = cmd[0:3]
        visualizer.add_sphere(
            center=p_ref,
            radius=0.018,
            color=(1.0, 0.55, 0.0, 0.95),
            label="ref",
        )


@dataclass(kw_only=True)
class DeterministicEETrackingCommandCfg(CommandTermCfg):
    trajectory_factory: TrajectoryFactory | None = None
    """Zero-arg callable returning a fresh trajectory object. Set via
    :meth:`set_trajectory` from the runner script."""

    lookahead_steps: int = 5
    lookahead_dt: float = 0.05
    asset_name: str = "robot"
    body_name: str = "hand"

    # Keep the base class' resample timer from ever firing mid-eval.
    resampling_time_range: tuple[float, float] = field(default=(1.0e9, 1.0e9))

    # Ghost-trajectory rendering. Drawn whenever the viewer enables debug
    # visualisation on the command term.
    debug_vis: bool = True
    ghost_samples: int = 120

    def set_trajectory(self, factory: TrajectoryFactory) -> None:
        self.trajectory_factory = factory

    def build(self, env: "ManagerBasedRlEnv") -> DeterministicEETrackingCommand:
        return DeterministicEETrackingCommand(self, env)
