from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

_AXIS_LUT = {"x": 0, "y": 1, "z": 2}


def _axis_basis(axes: tuple[str, str], device, dtype) -> torch.Tensor:
    e = torch.zeros(2, 3, device=device, dtype=dtype)
    for i, ax in enumerate(axes):
        if ax not in _AXIS_LUT:
            raise ValueError(f"axis must be one of x/y/z, got {ax}")
        e[i, _AXIS_LUT[ax]] = 1.0
    return e


def quintic_smoothstep(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def quintic_smoothstep_deriv(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    one_minus_u = 1.0 - u
    return 30.0 * u * u * one_minus_u * one_minus_u


@dataclass
class TrajectoryAnchor:

    pos: torch.Tensor
    quat: torch.Tensor


class _BaseDeterministicTraj:

    def __init__(self, T_total: float, n_cycles: float):
        if T_total <= 0:
            raise ValueError(f"T_total must be > 0, got {T_total}")
        self.T_total = float(T_total)
        self.n_cycles = float(n_cycles)
        self.s_max = 2.0 * torch.pi * self.n_cycles

    def set_anchor(self, anchor: TrajectoryAnchor) -> None:
        self._anchor = anchor

    def _phase(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = t / self.T_total
        s = self.s_max * quintic_smoothstep(u)
        s_dot = (self.s_max / self.T_total) * quintic_smoothstep_deriv(u)
        return s, s_dot

    def _orientation(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = t.shape[0]
        q = (
            self._anchor.quat.expand(N, 4)
            if self._anchor.quat.shape[0] == 1
            else (self._anchor.quat)
        )
        omega = torch.zeros(N, 3, device=t.device, dtype=t.dtype)
        return q, omega


class CircleTrajectory(_BaseDeterministicTraj):
    def __init__(
        self,
        radius: float = 0.15,
        axes: tuple[str, str] = ("x", "y"),
        T_total: float = 8.0,
        n_cycles: float = 1.0,
    ):
        super().__init__(T_total=T_total, n_cycles=n_cycles)
        self.radius = float(radius)
        self.axes = axes

    def evaluate(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = t.shape[0]
        device, dtype = t.device, t.dtype
        basis = _axis_basis(self.axes, device, dtype)  # (2, 3)
        u_hat, v_hat = basis[0], basis[1]

        s, s_dot = self._phase(t)  # (N,), (N,)
        cos_s = torch.cos(s)
        sin_s = torch.sin(s)

        # p(s) - anchor.pos = r * ((cos s - 1) u_hat + sin s v_hat)
        offset = self.radius * (
            (cos_s - 1.0).unsqueeze(-1) * u_hat + sin_s.unsqueeze(-1) * v_hat
        )  # (N, 3)
        anchor_pos = self._anchor.pos
        if anchor_pos.shape[0] == 1 and N > 1:
            anchor_pos = anchor_pos.expand(N, 3)
        p = anchor_pos + offset

        # v(s) = r * s_dot * (-sin s u_hat + cos s v_hat)
        v = (self.radius * s_dot).unsqueeze(-1) * (
            -sin_s.unsqueeze(-1) * u_hat + cos_s.unsqueeze(-1) * v_hat
        )

        q, omega = self._orientation(t)
        return p, v, q, omega


class LemniscateTrajectory(_BaseDeterministicTraj):
    def __init__(
        self,
        scale: float = 0.18,
        axes: tuple[str, str] = ("y", "z"),
        T_total: float = 10.0,
        n_cycles: float = 1.0,
    ):
        super().__init__(T_total=T_total, n_cycles=n_cycles)
        self.scale = float(scale)
        self.axes = axes

    def evaluate(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = t.shape[0]
        device, dtype = t.device, t.dtype
        basis = _axis_basis(self.axes, device, dtype)
        u_hat, v_hat = basis[0], basis[1]

        s, s_dot = self._phase(t)
        sin_s = torch.sin(s)
        cos_s = torch.cos(s)
        sin_2s = torch.sin(2.0 * s)
        cos_2s = torch.cos(2.0 * s)

        offset = self.scale * (
            sin_s.unsqueeze(-1) * u_hat + (0.5 * sin_2s).unsqueeze(-1) * v_hat
        )
        anchor_pos = self._anchor.pos
        if anchor_pos.shape[0] == 1 and N > 1:
            anchor_pos = anchor_pos.expand(N, 3)
        p = anchor_pos + offset

        # dp/ds = scale * (cos s u_hat + cos 2s v_hat); v = dp/ds * s_dot
        v = (self.scale * s_dot).unsqueeze(-1) * (
            cos_s.unsqueeze(-1) * u_hat + cos_2s.unsqueeze(-1) * v_hat
        )

        q, omega = self._orientation(t)
        return p, v, q, omega


def build_trajectory(
    kind: str,
    *,
    T_total: float,
    n_cycles: float = 1.0,
    radius: float = 0.15,
    scale: float = 0.18,
    axes: Sequence[str] | None = None,
) -> _BaseDeterministicTraj:
    if kind == "circle":
        ax = tuple(axes) if axes is not None else ("x", "y")
        return CircleTrajectory(
            radius=radius, axes=ax, T_total=T_total, n_cycles=n_cycles
        )
    if kind == "figure8":
        ax = tuple(axes) if axes is not None else ("y", "z")
        return LemniscateTrajectory(
            scale=scale, axes=ax, T_total=T_total, n_cycles=n_cycles
        )
    raise ValueError(f"unknown trajectory kind: {kind!r}")
