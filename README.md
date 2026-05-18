# EE Tracking on a Franka Panda — RL with sim-to-real-grade uncertainty

A reinforcement-learning system that controls a 7-DOF Franka Panda arm to track
arbitrary 3-D Cartesian trajectories, with position **and** orientation
tracking, smooth motion, and built-in robustness to sensor noise, action delay,
and dynamics mismatch.

Submission for the Humanoid (thehumanoid.ai) summer-internship challenge,
built on top of [mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated
MuJoCo Warp + Isaac-Lab-style manager API) and the integrated RSL-RL PPO
trainer.

---

## Quick start

```bash
uv sync                                                # install
uv run train Hac2k26-EETracking-Franka --num_envs 4096 # train
python -m hac2k26.eval.run \                            # evaluate
    --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_<iter>.pt \
    --output-dir eval_outputs/<run_tag> \
    --trajectories circle figure8
```

`<run>` and `<iter>` are auto-generated; pick the latest checkpoint under
`logs/rsl_rl/franka_ee_tracking/`. Outputs live in `eval_outputs/<run_tag>/`.

## What it does

| | |
|---|---|
| **Arm** | Franka Panda (7-DOF) |
| **Reference** | Randomised quintic-spline waypoint trajectories during training; deterministic circle / figure-8 during evaluation, anchored to the EE pose at reset |
| **Algorithm** | PPO with **asymmetric actor–critic** (actor sees noisy + delayed observations; critic sees ground-truth) |
| **Action** | Δ joint position (smooth by construction; zero output = hold pose) |
| **Sources of uncertainty** | Observation noise (Gaussian on EE & joint state), stochastic action delay (1–5 control steps), domain randomization (link inertia + joint friction) — re-sampled on every episode reset |
| **Orientation** | Tracked alongside position; encoded as 6-D rotation in observations, geodesic angle in reward |

## Results

Three numbers, one figure per trajectory. Deterministic eval, no DR
(`--disable-dr`), full sim-to-real uncertainty stack otherwise active.

| Trajectory | Position RMSE | EE jerk RMS | Orientation RMSE |
|---|---:|---:|---:|
| Circle (r = 15 cm, 1 cycle in 8 s) | _TBD_ mm | _TBD_ m/s³ | _TBD_ ° |
| Figure-8 (scale = 18 cm, 1 cycle in 8 s) | _TBD_ mm | _TBD_ m/s³ | _TBD_ ° |
| OSC baseline (damped-LS Jacobian + FF) — circle | _TBD_ mm | _TBD_ m/s³ | _TBD_ ° |

Per-trajectory plots: `eval_outputs/<run_tag>/<trajectory>/tracking.png`
— a 3-panel figure with the 3-D overlay, ||error|| timeseries, and EE jerk.

## Design choices

### State (actor observation)

Concatenation, noisy where indicated:

- Proprioception: joint position (±0.03 rad noise), joint velocity (±0.1 rad/s),
  last action.
- EE state (world frame): position (±5 mm), linear velocity (±5 cm/s),
  angular velocity (±0.1 rad/s), 6-D rotation (±0.01).
- Reference: target position, velocity, orientation (as 6-D rotation), and a
  phase variable φ ∈ [0, 1] giving the policy explicit knowledge of where it
  is in the trajectory cycle.
- **Lookahead window**: 5 future reference positions at 50 ms spacing —
  lets the policy *anticipate* turning points instead of merely reacting.
- **Explicit tracking errors** (`p_ref − p_ee`, `v_ref − v_ee`, rotvec error,
  ω error), with the **same noise magnitude** as the corresponding raw EE
  observation — so the actor cannot bypass sensor noise by reading the
  (otherwise clean) error channel. The privileged critic still sees the
  clean errors.
- **Manipulability** w(q) = √det(J·Jᵀ) — gives the policy direct
  awareness of kinematic singularities.

### Action

`Δ joint position` ∈ [−0.1, +0.1] rad, applied additively at every 50 Hz
control step on top of the current joint angles. Smoothness is built in:
zero output = hold pose, no DC offset to fight. The action is then routed
through a `DelayBuffer` that serves a frame from the past with a lag sampled
uniformly in {1, …, 5} control steps per environment, per step.

### Reward

```
r =  w_pos  · two-scale Gaussian on ||p_ee − p_ref||           (coarse 30, fine 400)
   + w_ori  · two-scale Gaussian on geodesic orientation err   (coarse 2,  fine 20)
   + w_vel  · v_ee · v̂_ref                                     (tangent alignment)
   − w_l2   · ||p_ee − p_ref||                                  (always-on dense pull)
   − w_rate · ||a_t − a_{t-1}||²                                (1st-order smoothness)
   − w_jerk · ||a_t − 2·a_{t-1} + a_{t-2}||²                    (2nd-order smoothness)
   − w_sing · max(0, w_min − w(q))²                             (one-sided singularity barrier)
```

The two-scale Gaussian gives a usable gradient both far from the target
(coarse kernel) and at sub-cm precision (fine kernel). The jerk penalty was
the single most effective regulariser against high-frequency action jitter.

### Trajectory representation

Two sampling regimes share the same observation interface so the same actor
can be trained, played-back, and evaluated under both:

- **Global cylindrical** (default): waypoints are sampled in (r, θ, z) with
  θ accumulated as a bounded random walk from the current EE angle, then
  reconstructed as Cartesian via cos/sin. Hard-clamped to ±2.5 rad to stay
  inside joint-1's soft position limit (good for large angular sweeps
  centred on the base axis).
- **Local cartesian**: waypoints sampled in a ±radius cube around the EE,
  clamped to a workspace box (good for small circles / figure-8s near the
  current EE pose).

Both fit a **quintic-Hermite spline** through the waypoints with zero
acceleration at every waypoint (C² across segment boundaries) — smooth by
construction, no velocity discontinuities to chase. A curriculum slowly
widens the sampling radius (or Δθ) over the first ~3300 PPO iterations.

The orientation reference is generated independently: random waypoint
quaternions perturbed from the current EE, interpolated segment-by-segment
via a SLERP-equivalent rest-to-rest profile (quintic smoothstep on the
axis-angle log).

### Evaluation methodology

`hac2k26.eval.run` swaps the random training command for a deterministic
analytic trajectory (circle or figure-8) anchored to the current EE pose,
runs the trained policy for a fixed horizon, and reports **three** headline
numbers:

1. **Position RMSE [mm]** — accuracy over time.
2. **EE jerk RMS [m/s³]** — smoothness, computed via numerical third
   derivatives of the recorded EE position.
3. **Orientation RMSE [deg]** — geodesic angle between target and actual
   EE orientation, RMS over the episode.

A single `tracking.png` per trajectory gives the visual story: 3-D overlay
on top, ||tracking error|| in the middle, ||jerk|| at the bottom.

Three flags worth knowing:

- `--controller {rl,osc}` — swap in a classical operational-space-control
  baseline (damped-least-squares Jacobian inverse + optional velocity
  feedforward). Useful to quantify *what* the RL policy buys you, and on
  *which* trajectories.
- `--disable-action-delay` — drops the delay buffer so you can measure the
  pure controller-tracking floor (vs. the delay-induced lag).
- `--disable-dr` — disables the randomized inertia and joint-friction
  events on env reset for deterministic, reproducible eval numbers. By
  default the eval env keeps the same DR distribution as training (each
  rollout draws a fresh dynamics sample), which is more honest for
  measuring robustness but adds variance to the headline numbers.

### Why those uncertainty sources

The challenge requires *at least one*. We include three because each tests
a different failure mode of the controller:

- **Observation noise** (Uniform, applied per term) — sensor-grade
  measurement noise; the policy learns to filter via history embedded in
  joint velocities and consecutive observations.
- **Stochastic action delay** (1–5 control steps) — control-stack latency;
  the policy must extrapolate forward, which is exactly what the lookahead
  window enables.
- **Domain randomization** (link inertia ±15 %, joint friction 0–0.2 N·m,
  re-sampled every episode reset) — model mismatch; trains a policy that
  generalises beyond a single tuned dynamics.

## Repository layout

```
src/hac2k26/
├── assets/robots/franka_panda/      # Panda actuator & home-pose config
├── ee_tracking/
│   ├── config/franka/               # Task registration + RL hyperparams
│   ├── tracking_env_cfg.py          # Env factory: obs / actions / rewards / events
│   └── mdp/
│       ├── actions.py               # Delayed Δ-joint action
│       ├── commands.py              # Random quintic spline command (training)
│       ├── events.py                # Reset to home pose
│       ├── kinematics.py            # Manipulability + singularity penalty
│       ├── observations.py          # Actor / critic observation terms
│       └── rewards.py               # Two-scale tracking, smoothness, singularity
├── trajectory/quintic_spline.py     # Quintic-Hermite spline primitive
├── baseline/osc_controller.py       # Classical OSC controller for comparison
└── eval/
    ├── trajectories.py              # Deterministic circle / figure-8
    ├── commands.py                  # Deterministic command term
    ├── metrics.py                   # RMSE / jerk RMS / orientation RMSE
    ├── plotting.py                  # Single 3-panel tracking figure
    └── run.py                       # CLI eval runner

scripts/
├── eval_play.py                     # Live viewer of a trained policy
├── play_franka.py                   # Visualise the env with zero actions
└── plot_spline.py                   # Spline sanity check
```

## Notes

- Trained for ~3.3 k PPO iterations (~10 minutes on a single L4 / A100)
  with 4096 parallel environments under the full uncertainty stack.
- All hyper-parameters in `src/hac2k26/ee_tracking/config/franka/rl_cfg.py`.
- Trajectory hyper-parameters in `src/hac2k26/ee_tracking/tracking_env_cfg.py`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
