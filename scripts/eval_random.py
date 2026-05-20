from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import hac2k26  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

DEFAULT_TASK_ID = "Hac2k26-EETracking-Franka"


def _build_env(task_id: str, num_envs: int, device: str) -> ManagerBasedRlEnv:
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = num_envs
    env_cfg.terminations = {}
    return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _rollout(env, policy, n_steps: int, body_id: int):
    asset = env.unwrapped.scene["robot"]
    cm = env.unwrapped.command_manager
    N = env.unwrapped.num_envs
    dev = env.unwrapped.device

    p_ee = torch.zeros(n_steps, N, 3, device=dev)
    q_ee = torch.zeros(n_steps, N, 4, device=dev)
    p_ref = torch.zeros(n_steps, N, 3, device=dev)
    q_ref = torch.zeros(n_steps, N, 4, device=dev)

    obs, _ = env.reset()
    with torch.inference_mode():
        for k in range(n_steps):
            p_ee[k] = asset.data.body_link_pos_w[:, body_id, :]
            q_ee[k] = asset.data.body_link_quat_w[:, body_id, :]
            cmd = cm.get_command("trajectory")
            p_ref[k] = cmd[:, 0:3]
            q_ref[k] = cmd[:, 7:11]
            action = policy(obs)
            obs, _, _, _ = env.step(action)

    return (
        p_ee.cpu().numpy(),
        q_ee.cpu().numpy(),
        p_ref.cpu().numpy(),
        q_ref.cpu().numpy(),
    )


def _per_env_metrics(p_ee, q_ee, p_ref, q_ref, dt):
    err = p_ref - p_ee
    pos_rmse = np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=0))  # (N,)

    v = np.gradient(p_ee, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    jerk_mag = np.linalg.norm(j, axis=-1)  # (T, N)
    jerk_rms = np.sqrt(np.mean(jerk_mag * jerk_mag, axis=0))  # (N,)

    dot = np.abs(np.sum(q_ee * q_ref, axis=-1)).clip(-1.0, 1.0)
    ang = 2.0 * np.arccos(dot)  # (T, N)
    orient_rmse = np.sqrt(np.mean(ang * ang, axis=0))  # (N,)

    return pos_rmse, jerk_rms, orient_rmse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK_ID)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env = _build_env(args.task, args.num_envs, device)
    body_id = env.scene["robot"].body_names.index("hand")
    step_dt = env.step_dt
    n_steps = int(round(env.cfg.episode_length_s / step_dt))

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

    print(
        f"Rolling out {n_steps} steps × {args.num_envs} envs "
        f"({n_steps * step_dt:.1f}s per env)..."
    )
    p_ee, q_ee, p_ref, q_ref = _rollout(wrapped, policy, n_steps, body_id)
    wrapped.close()

    pos_rmse, jerk_rms, orient_rmse = _per_env_metrics(
        p_ee, q_ee, p_ref, q_ref, step_dt
    )

    def _row(name, x, unit, scale=1.0):
        x = x * scale
        return (
            name,
            float(np.mean(x)),
            float(np.std(x)),
            float(np.percentile(x, 50)),
            float(np.percentile(x, 95)),
            float(np.max(x)),
            unit,
        )

    rows = [
        _row("Position RMSE", pos_rmse, "mm", scale=1e3),
        _row("EE jerk RMS", jerk_rms, "m/s³"),
        _row("Orient RMSE", orient_rmse, "deg", scale=180.0 / np.pi),
    ]

    print()
    print(
        f"Random training-distribution eval "
        f"({args.num_envs} envs x {n_steps * step_dt:.1f}s):"
    )
    print(
        f"{'metric':16s} {'mean':>10s} {'std':>10s} "
        f"{'p50':>10s} {'p95':>10s} {'max':>10s}  unit"
    )
    print("-" * 80)
    for name, mean, std, p50, p95, mx, unit in rows:
        print(
            f"{name:16s} {mean:10.2f} {std:10.2f} "
            f"{p50:10.2f} {p95:10.2f} {mx:10.2f}  {unit}"
        )


if __name__ == "__main__":
    main()
