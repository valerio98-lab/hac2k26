"""Plot helper for the eval suite.

Single function ``plot_tracking`` emits a 3-panel summary figure
(3D overlay, error-norm over time, EE jerk over time) for one
deterministic rollout. The headline of the eval pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — side effect: 3D proj

from hac2k26.eval.metrics import ee_jerk, position_error_norm


def plot_tracking(
    t: np.ndarray,
    p_ee: np.ndarray,
    p_ref: np.ndarray,
    dt: float,
    title: str,
    out: Path,
) -> None:
    """One PNG with 3 stacked panels: 3D overlay, ||err||, ||jerk||.

    Args:
        t: time vector, shape (T,).
        p_ee: actual EE positions, shape (T, 3).
        p_ref: reference positions, shape (T, 3).
        dt: control step in seconds (for numerical jerk).
        title: figure suptitle.
        out: PNG path.
    """
    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=(3.0, 1.4, 1.4), hspace=0.35)

    # 3D overlay.
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.plot(
        p_ref[:, 0],
        p_ref[:, 1],
        p_ref[:, 2],
        color="C0",
        lw=2.2,
        label="reference",
    )
    ax3d.plot(
        p_ee[:, 0],
        p_ee[:, 1],
        p_ee[:, 2],
        color="C3",
        lw=1.3,
        alpha=0.9,
        label="EE",
    )
    ax3d.scatter(
        p_ref[0, 0],
        p_ref[0, 1],
        p_ref[0, 2],
        color="k",
        s=30,
        marker="o",
        label="start",
    )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(loc="upper left", fontsize=9)

    # Tracking error magnitude.
    ax_err = fig.add_subplot(gs[1])
    err_mm = position_error_norm(p_ee, p_ref) * 1e3
    ax_err.plot(t, err_mm, color="C3", lw=1.4)
    ax_err.set_ylabel("||p_ee - p_ref|| [mm]")
    ax_err.grid(True, alpha=0.3)

    # End-effector jerk magnitude.
    ax_jerk = fig.add_subplot(gs[2], sharex=ax_err)
    j = ee_jerk(p_ee, dt)
    ax_jerk.plot(t, j, color="C2", lw=1.2)
    ax_jerk.set_xlabel("time [s]")
    ax_jerk.set_ylabel("||jerk|| [m/s³]")
    ax_jerk.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
