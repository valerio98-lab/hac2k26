"""
Examples:

    uv run python scripts/eval_play.py \\
        --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_2999.pt \\
        --trajectory circle --duration 8.0 --radius 0.15

    uv run python scripts/eval_play.py \\
        --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_2999.pt \\
        --trajectory figure8 --duration 8.0 --scale 0.18
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

import hac2k26  # noqa: F401 — registers Hac2k26-EETracking-Franka

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from hac2k26.eval.run import DEFAULT_TASK_ID, _build_eval_env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--task",
        default=DEFAULT_TASK_ID,
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--trajectory", default="circle", choices=["circle", "figure8"])
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--n-cycles", type=float, default=1.0)
    p.add_argument(
        "--radius",
        type=float,
        default=0.15,
        help="Circle radius (m). Ignored for figure8.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.18,
        help="Lemniscate half-width (m). Ignored for circle.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--viewer", default="auto", choices=["auto", "native", "viser"])
    p.add_argument("--disable-action-delay", action="store_true")
    p.add_argument(
        "--disable-dr",
        action="store_true",
        help="Disable domain randomization for deterministic playback.",
    )
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = _build_eval_env(
        task_id=args.task,
        traj_kind=args.trajectory,
        duration=args.duration,
        n_cycles=args.n_cycles,
        radius=args.radius,
        scale=args.scale,
        device=device,
        disable_action_delay=args.disable_action_delay,
        disable_dr=args.disable_dr,
    )

    # Long episode_length_s means the trajectory plays once, then the env
    # holds the final reference (constant) until time_out. For continuous
    # playback we'd want a periodic trajectory; for now the user can hit
    # the viewer's reset button to replay.
    agent_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    if args.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        chosen = "native" if has_display else "viser"
    else:
        chosen = args.viewer
    print(
        f"[INFO]: trajectory={args.trajectory}  viewer={chosen}  "
        f"duration={args.duration}s  n_cycles={args.n_cycles}"
    )

    if chosen == "native":
        NativeMujocoViewer(wrapped, policy).run()
    else:
        ViserPlayViewer(wrapped, policy).run()

    wrapped.close()


if __name__ == "__main__":
    main()
