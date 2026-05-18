from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def position_error(p_ee: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Signed per-axis error e = p_ref - p_ee. Shape (T, 3)."""
    return p_ref - p_ee


def position_error_norm(p_ee: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """||p_ref - p_ee||_2 over time. Shape (T,)."""
    return np.linalg.norm(p_ref - p_ee, axis=-1)


def position_rmse(p_ee: np.ndarray, p_ref: np.ndarray) -> dict[str, float]:
    """Per-axis and overall RMSE.

    Returns keys: x, y, z, total (Euclidean RMS of L2 error).
    """
    err = position_error(p_ee, p_ref)
    per_axis = np.sqrt(np.mean(err * err, axis=0))
    total = float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))
    return {
        "x": float(per_axis[0]),
        "y": float(per_axis[1]),
        "z": float(per_axis[2]),
        "total": total,
    }


def ee_jerk(p_ee: np.ndarray, dt: float) -> np.ndarray:
    """Numerical EE jerk (m/s^3) via central differences on the third
    derivative. Shape (T,) the L2 norm of the per-axis jerk.
    """
    if p_ee.shape[0] < 4:
        return np.zeros(p_ee.shape[0])
    v = np.gradient(p_ee, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    return np.linalg.norm(j, axis=-1)


def jerk_summary(p_ee: np.ndarray, dt: float) -> dict[str, float]:
    j = ee_jerk(p_ee, dt)
    return {
        "mean": float(np.mean(j)),
        "rms": float(np.sqrt(np.mean(j * j))),
        "max": float(np.max(j)),
    }


def orientation_angle_error(q_ee: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between EE and reference quaternions.

    Both inputs are (T, 4) in wxyz convention.
    """
    dot = np.abs(np.sum(q_ee * q_ref, axis=-1)).clip(-1.0, 1.0)
    return 2.0 * np.arccos(dot)


@dataclass
class RolloutMetrics:
    """
    Positional RMSE, EE jerk smoothness, orientation RMSE.
    """

    rmse_pos_total: float  # meters
    jerk_rms: float  # m/s^3
    orient_rmse_rad: float

    def to_flat_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}rmse_pos_total": self.rmse_pos_total,
            f"{prefix}jerk_rms": self.jerk_rms,
            f"{prefix}orient_rmse_rad": self.orient_rmse_rad,
        }


def compute_all(
    p_ee: np.ndarray,
    p_ref: np.ndarray,
    q_ee: np.ndarray,
    q_ref: np.ndarray,
    dt: float,
) -> RolloutMetrics:
    rmse = position_rmse(p_ee, p_ref)
    j = jerk_summary(p_ee, dt)
    ang = orientation_angle_error(q_ee, q_ref)
    orient_rmse = float(np.sqrt(np.mean(ang * ang)))
    return RolloutMetrics(
        rmse_pos_total=rmse["total"],
        jerk_rms=j["rms"],
        orient_rmse_rad=orient_rmse,
    )
