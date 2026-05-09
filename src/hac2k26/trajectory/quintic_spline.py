"""Quintic spline trajectory math.

Pure PyTorch. Operates on vector waypoints (typical use: 3D EE positions). The
piecewise spline chains quintic segments — each segment is a degree-5
polynomial parameterized by boundary position, velocity and acceleration. Six
boundary conditions per segment determine six coefficients in closed form.

Coefficients live in a (6, D) tensor: rows c0..c5, columns the spatial dims.

Device convention: tensors stay on the device of the inputs. Pass CPU tensors
in plot/test scripts; pass GPU tensors when integrating into the command.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def quintic_coeffs(
    p0: torch.Tensor,
    p1: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """Closed-form quintic coefficients for one segment t in [0, T].

    The polynomial p(t) = c0 + c1·t + c2·t² + c3·t³ + c4·t⁴ + c5·t⁵ matches
    the boundary conditions:
        p(0)=p0, p(T)=p1, p'(0)=v0, p'(T)=v1, p''(0)=a0, p''(T)=a1. (De Luca docet)

    Returns (6, D).
    """
    if T <= 0.0:
        raise ValueError(f"Segment duration must be positive, got T={T}")

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    dp = p1 - p0

    c0 = p0
    c1 = v0
    c2 = a0 / 2.0
    c3 = (20.0 * dp - (8.0 * v1 + 12.0 * v0) * T - (3.0 * a0 - a1) * T2) / (2.0 * T3)
    c4 = (-30.0 * dp + (14.0 * v1 + 16.0 * v0) * T + (3.0 * a0 - 2.0 * a1) * T2) / (
        2.0 * T4
    )
    c5 = (12.0 * dp - 6.0 * (v0 + v1) * T - (a0 - a1) * T2) / (2.0 * T5)

    return torch.stack([c0, c1, c2, c3, c4, c5], dim=0)


def evaluate_segment(
    coeffs: torch.Tensor,
    t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one quintic segment at time(s) t.

    Args:
        coeffs: (6, D) from `quintic_coeffs`.
        t: Python scalar or 1-D tensor of local times within [0, T].

    Returns:
        (pos, vel, acc). Shapes: (D,) if t is scalar, else (N, D).
    """
    device, dtype = coeffs.device, coeffs.dtype

    if isinstance(t, torch.Tensor):
        scalar = t.dim() == 0
        t_tensor = t.to(device=device, dtype=dtype)
        if scalar:
            t_tensor = t_tensor.unsqueeze(0)
    else:
        scalar = True
        t_tensor = torch.tensor([float(t)], device=device, dtype=dtype)

    # Power matrix: row i = t^i, shape (6, N).
    powers = torch.stack([t_tensor**i for i in range(6)], dim=0)

    # Position. powers.T: (N, 6); coeffs: (6, D).
    pos = powers.T @ coeffs  # (N, D)

    # Derivative coefficients
    vel_coeffs = torch.stack(
        [(i + 1) * coeffs[i + 1] for i in range(5)], dim=0
    )  # (5, D)
    vel = powers[:5].T @ vel_coeffs  # (N, D)

    acc_coeffs = torch.stack(
        [(i + 1) * (i + 2) * coeffs[i + 2] for i in range(4)], dim=0
    )  # (4, D)
    acc = powers[:4].T @ acc_coeffs  # (N, D)

    if scalar:
        return pos[0], vel[0], acc[0]
    return pos, vel, acc


@dataclass
class QuinticSpline:
    """Piecewise quintic spline.

    Attributes:
        segments: shape (n_segments, 6, D) — coefficients per segment.
        cumulative_times: shape (n_segments + 1,). Index i is the global start
            time of segment i; the last entry is the total duration.
    """

    segments: torch.Tensor  # (S, 6, D)
    cumulative_times: torch.Tensor  # (S+1,)

    @property
    def total_duration(self) -> float:
        return float(self.cumulative_times[-1].item())

    @property
    def num_segments(self) -> int:
        return int(self.segments.shape[0])

    @property
    def device(self) -> torch.device:
        return self.segments.device

    @property
    def dtype(self) -> torch.dtype:
        return self.segments.dtype


def build_rest_to_rest_spline(
    waypoints: torch.Tensor,
    segment_durations: float | torch.Tensor,
) -> QuinticSpline:
    """Build a spline through `waypoints` with v=0, a=0 at every waypoint.

    Args:
        waypoints: shape (K, D), K >= 2. K-1 segments will be built.
        segment_durations: per-segment duration. Python scalar or 1-D tensor of
            length K-1.
    """
    if waypoints.dim() != 2 or waypoints.shape[0] < 2:
        raise ValueError(f"waypoints must be (K, D) with K>=2, got {waypoints.shape}")

    device, dtype = waypoints.device, waypoints.dtype
    K = waypoints.shape[0]
    n_seg = K - 1

    if isinstance(segment_durations, torch.Tensor):
        durations = segment_durations.to(device=device, dtype=dtype)
        if durations.dim() == 0:
            durations = durations.expand(n_seg).clone()
        if durations.shape != (n_seg,):
            raise ValueError(
                f"segment_durations must broadcast to ({n_seg},), got {durations.shape}"
            )
    else:
        durations = torch.full(
            (n_seg,), float(segment_durations), device=device, dtype=dtype
        )

    if torch.any(durations <= 0):
        raise ValueError("All segment durations must be positive.")

    zero = torch.zeros_like(waypoints[0])
    segs = torch.stack(
        [
            quintic_coeffs(
                waypoints[i],
                waypoints[i + 1],
                zero,
                zero,
                zero,
                zero,
                float(durations[i].item()),
            )
            for i in range(n_seg)
        ],
        dim=0,
    )  # (n_seg, 6, D)

    cumulative = torch.cat(
        [torch.zeros(1, device=device, dtype=dtype), torch.cumsum(durations, dim=0)]
    )
    return QuinticSpline(segments=segs, cumulative_times=cumulative)


def evaluate_spline(
    spline: QuinticSpline,
    t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the spline at global time(s) t.

    Times are clamped to [0, total_duration]. Returns (pos, vel, acc), each of
    shape (D,) for scalar t or (N, D) for a 1-D tensor of times.
    """
    device, dtype = spline.device, spline.dtype

    if isinstance(t, torch.Tensor):
        scalar = t.dim() == 0
        t_tensor = t.to(device=device, dtype=dtype)
        if scalar:
            t_tensor = t_tensor.unsqueeze(0)
    else:
        scalar = True
        t_tensor = torch.tensor([float(t)], device=device, dtype=dtype)

    t_tensor = t_tensor.clamp(0.0, spline.total_duration)

    # Find segment index for each t. Search interior boundaries; clamp the
    # trailing endpoint to the last segment.
    interior = spline.cumulative_times[1:-1]  # (n_seg - 1,)
    seg_idx = torch.searchsorted(interior, t_tensor, right=True)
    seg_idx = seg_idx.clamp(0, spline.num_segments - 1)  # (N,)

    # Per-time local time and per-time coefficients.
    local_t = t_tensor - spline.cumulative_times[seg_idx]  # (N,)
    selected = spline.segments[seg_idx]  # (N, 6, D)

    # Vectorised polynomial evaluation. powers[n, i] = local_t[n]^i.
    powers = torch.stack([local_t**i for i in range(6)], dim=-1)  # (N, 6)

    pos = torch.einsum("ni,nij->nj", powers, selected)  # (N, D)

    vel_coeffs = torch.stack(
        [(i + 1) * selected[..., i + 1, :] for i in range(5)], dim=-2
    )  # (N, 5, D)
    vel = torch.einsum("ni,nij->nj", powers[..., :5], vel_coeffs)

    acc_coeffs = torch.stack(
        [(i + 1) * (i + 2) * selected[..., i + 2, :] for i in range(4)], dim=-2
    )  # (N, 4, D)
    acc = torch.einsum("ni,nij->nj", powers[..., :4], acc_coeffs)

    if scalar:
        return pos[0], vel[0], acc[0]
    return pos, vel, acc
