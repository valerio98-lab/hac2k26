"""CLI entry point for deterministic EE-tracking evaluation.

Loads a trained checkpoint, swaps the random quintic-spline command term
for a deterministic circle / figure-8 trajectory, rolls out a fixed
horizon and writes per-step CSV + summary PNG plots.

Usage:
    uv run python -m hac2k26.eval.run \\
        --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_2999.pt \\
        --output-dir eval_outputs/<run_tag> \\
        --trajectories circle figure8 \\
        --duration 8.0 --n-cycles 1.0 --radius 0.15 --scale 0.18

Outputs (per trajectory):
    eval_outputs/<run_tag>/<traj_name>/rollout.csv
    eval_outputs/<run_tag>/<traj_name>/tracking.png  (3-panel: 3D + err + jerk)
    eval_outputs/<run_tag>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import hac2k26  # noqa: F401 — registers Hac2k26-EETracking-Franka

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from hac2k26.baseline.osc_controller import OSCController
from hac2k26.eval import metrics as eval_metrics
from hac2k26.eval import plotting
from hac2k26.eval.commands import DeterministicEETrackingCommandCfg
from hac2k26.eval.trajectories import build_trajectory

DEFAULT_TASK_ID = "Hac2k26-EETracking-Franka"


def _trajectory_factory(kind: str, **kw):
    """Return a zero-arg callable producing a fresh trajectory."""
    return partial(build_trajectory, kind, **kw)


def _build_eval_env(
    *,
    task_id: str,
    traj_kind: str,
    duration: float,
    n_cycles: float,
    radius: float,
    scale: float,
    device: str,
    disable_action_delay: bool,
    disable_dr: bool = False,
):
    """Construct the env configured for deterministic playback."""
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1

    env_cfg.episode_length_s = float(duration) + 1.0
    env_cfg.terminations = {}

    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False

    if disable_action_delay:
        action_cfg = env_cfg.actions["joint_pos"]
        action_cfg.min_lag = 0
        action_cfg.max_lag = 0

    if disable_dr:
        for term_name in ("dr_pseudo_inertia", "dr_joint_friction"):
            if term_name in env_cfg.events:
                env_cfg.events[term_name] = None

    train_cmd = env_cfg.commands["trajectory"]
    factory = _trajectory_factory(
        traj_kind,
        T_total=float(duration),
        n_cycles=float(n_cycles),
        radius=float(radius),
        scale=float(scale),
    )
    env_cfg.commands["trajectory"] = DeterministicEETrackingCommandCfg(
        trajectory_factory=factory,
        lookahead_steps=train_cmd.lookahead_steps,
        lookahead_dt=train_cmd.lookahead_dt,
        asset_name=train_cmd.asset_name,
        body_name=train_cmd.body_name,
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    return env


def _load_policy(env, task_id: str, checkpoint: Path, device: str):
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return wrapped, runner.get_inference_policy(device=device)


def _rollout(env, policy, n_steps: int, body_id: int) -> dict[str, np.ndarray]:
    """Step the env for ``n_steps`` collecting reference + actual EE state."""
    asset = env.unwrapped.scene["robot"]
    cm = env.unwrapped.command_manager

    p_ee = np.zeros((n_steps, 3), dtype=np.float32)
    v_ee = np.zeros((n_steps, 3), dtype=np.float32)
    q_ee = np.zeros((n_steps, 4), dtype=np.float32)
    p_ref = np.zeros((n_steps, 3), dtype=np.float32)
    v_ref = np.zeros((n_steps, 3), dtype=np.float32)
    q_ref = np.zeros((n_steps, 4), dtype=np.float32)
    actions = np.zeros(
        (n_steps, env.unwrapped.action_space.shape[-1]), dtype=np.float32
    )

    obs, _ = env.reset()

    with torch.inference_mode():
        for k in range(n_steps):
            p_ee[k] = asset.data.body_link_pos_w[0, body_id, :].cpu().numpy()
            v_ee[k] = asset.data.body_link_lin_vel_w[0, body_id, :].cpu().numpy()
            q_ee[k] = asset.data.body_link_quat_w[0, body_id, :].cpu().numpy()
            cmd = cm.get_command("trajectory")[0]
            p_ref[k] = cmd[0:3].cpu().numpy()
            v_ref[k] = cmd[3:6].cpu().numpy()
            q_ref[k] = cmd[7:11].cpu().numpy()

            action = policy(obs)
            actions[k] = action[0].detach().cpu().numpy()
            obs, _, _, _ = env.step(action)

    return {
        "p_ee": p_ee,
        "v_ee": v_ee,
        "q_ee": q_ee,
        "p_ref": p_ref,
        "v_ref": v_ref,
        "q_ref": q_ref,
        "actions": actions,
    }


def _write_csv(path: Path, t: np.ndarray, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["t"]
    cols: list[np.ndarray] = [t]
    for name in ("p_ref", "p_ee"):
        for ax in ("x", "y", "z"):
            headers.append(f"{name}_{ax}")
        cols.extend([data[name][:, i] for i in range(3)])
    err_norm = np.linalg.norm(data["p_ref"] - data["p_ee"], axis=-1)
    headers.append("err_norm")
    cols.append(err_norm)

    arr = np.stack(cols, axis=1)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(arr.tolist())


def _run_one_trajectory(
    *,
    task_id: str,
    traj_kind: str,
    controller: str,
    checkpoint: Path | None,
    out_dir: Path,
    duration: float,
    n_cycles: float,
    radius: float,
    scale: float,
    device: str,
    disable_action_delay: bool,
    disable_dr: bool,
    osc_kp_pos: float,
    osc_kp_ori: float,
    osc_damping: float,
    osc_max_dq: float,
    osc_no_orientation: bool,
    osc_no_feedforward: bool,
) -> dict[str, float]:
    env = _build_eval_env(
        task_id=task_id,
        traj_kind=traj_kind,
        duration=duration,
        n_cycles=n_cycles,
        radius=radius,
        scale=scale,
        device=device,
        disable_action_delay=disable_action_delay,
        disable_dr=disable_dr,
    )
    body_id = env.scene["robot"].body_names.index("hand")
    step_dt = env.step_dt
    n_steps = int(round(duration / step_dt))

    agent_cfg = load_rl_cfg(task_id)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if controller == "rl":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --controller rl.")
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    elif controller == "osc":
        action_scale = float(env.cfg.actions["joint_pos"].scale)
        policy = OSCController(
            env,
            kp_pos=osc_kp_pos,
            kp_ori=0.0 if osc_no_orientation else osc_kp_ori,
            damping=osc_damping,
            max_dq=osc_max_dq,
            action_scale=action_scale,
            use_velocity_feedforward=not osc_no_feedforward,
        )
    else:
        raise ValueError(f"unknown controller: {controller!r}")

    data = _rollout(wrapped, policy, n_steps, body_id)
    wrapped.close()

    t = np.arange(n_steps, dtype=np.float32) * step_dt
    out_dir = out_dir / traj_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "rollout.csv", t, data)

    m = eval_metrics.compute_all(
        p_ee=data["p_ee"],
        p_ref=data["p_ref"],
        q_ee=data["q_ee"],
        q_ref=data["q_ref"],
        dt=step_dt,
    )

    title = f"{traj_kind} — {duration:.1f}s, n_cycles={n_cycles}"
    plotting.plot_tracking(
        t=t,
        p_ee=data["p_ee"],
        p_ref=data["p_ref"],
        dt=step_dt,
        title=title,
        out=out_dir / "tracking.png",
    )

    flat = m.to_flat_dict()
    print(
        f"[{traj_kind}] RMSE: {flat['rmse_pos_total'] * 1e3:.2f} mm   "
        f"jerk rms: {flat['jerk_rms']:.2f} m/s³   "
        f"orient rmse: {np.rad2deg(flat['orient_rmse_rad']):.2f}°"
    )
    return flat


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--task",
        default=DEFAULT_TASK_ID,
        help="Which registered task to load.",
    )
    p.add_argument(
        "--controller",
        default="rl",
        choices=["rl", "osc"],
        help="Which controller to evaluate. OSC = classical damped-LS Jacobian baseline.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None, help="Required when --controller rl."
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--trajectories",
        nargs="+",
        default=["circle", "figure8"],
        choices=["circle", "figure8"],
    )
    p.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Total trajectory duration in seconds.",
    )
    p.add_argument(
        "--n-cycles",
        type=float,
        default=1.0,
        help="Number of full cycles within the duration.",
    )
    p.add_argument("--radius", type=float, default=0.15, help="Circle radius (m).")
    p.add_argument(
        "--scale", type=float, default=0.18, help="Lemniscate half-width scale (m)."
    )
    p.add_argument(
        "--device",
        default=None,
    )
    p.add_argument(
        "--disable-action-delay",
        action="store_true",
        help="Drop the stochastic action delay during evaluation.",
    )
    p.add_argument(
        "--disable-dr",
        action="store_true",
        help=(
            "Disable domain randomization (mass / inertia / joint friction) "
            "for reproducible metrics"
        ),
    )
    # OSC tuning.
    p.add_argument("--osc-kp-pos", type=float, default=12.0)
    p.add_argument("--osc-kp-ori", type=float, default=4.0)
    p.add_argument("--osc-damping", type=float, default=0.05)
    p.add_argument("--osc-max-dq", type=float, default=0.1)
    p.add_argument(
        "--osc-no-orientation",
        action="store_true",
        help="Disable orientation tracking in OSC (position only).",
    )
    p.add_argument(
        "--osc-no-feedforward",
        action="store_true",
        help="Disable velocity feedforward in OSC.",
    )
    args = p.parse_args(argv)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    for kind in args.trajectories:
        flat = _run_one_trajectory(
            task_id=args.task,
            traj_kind=kind,
            controller=args.controller,
            checkpoint=args.checkpoint,
            out_dir=args.output_dir,
            duration=args.duration,
            n_cycles=args.n_cycles,
            radius=args.radius,
            scale=args.scale,
            device=device,
            disable_action_delay=args.disable_action_delay,
            disable_dr=args.disable_dr,
            osc_kp_pos=args.osc_kp_pos,
            osc_kp_ori=args.osc_kp_ori,
            osc_damping=args.osc_damping,
            osc_max_dq=args.osc_max_dq,
            osc_no_orientation=args.osc_no_orientation,
            osc_no_feedforward=args.osc_no_feedforward,
        )
        row: dict[str, float | str] = {
            "trajectory": kind,
            "controller": args.controller,
            "task": args.task,
        }
        row.update(flat)
        summary_rows.append(row)

    if summary_rows:
        keys = list(summary_rows[0].keys())
        out = args.output_dir / "summary.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\nWrote summary: {out}")


if __name__ == "__main__":
    main()
