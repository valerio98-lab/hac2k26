# EE Tracking on a Franka Panda — RL core component + sim2real

A reinforcement-learning system that controls a 7-DOF Franka Panda arm to track arbitrary 3-D Cartesian trajectories, with position and orientation tracking, smooth motion, and built-in robustness to sensor noise, action delay and dynamics mismatch.


Built on top of [mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo Warp + Isaac-Lab-style manager API) and the integrated RSL-RL PPO
trainer.

---

## Quick start

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or, if you don't have curl:
# wget -qO- https://astral.sh/uv/install.sh | sh
```

### Install the repository
```bash
git clone https://github.com/valerio98-lab/hac2k26.git
cd hac2k26
uv sync
```

### Train a policy
```bash
uv run train Hac2k26-EETracking-Franka --num_envs 4096
```

### Play a policy
```bash
uv run play Hac2k26-EETracking-Franka --checkpoint-file <path>
```

### Play the RL policy on a deterministic circle + figure-8
```bash
uv run play Hac2k26-EETracking-Franka --checkpoint-file <path>
```

## At a glance

A quick scan of what's inside, mapped to the challenge requirements. The detailed sections below dig into each item.

- **Simulated arm**: Franka Panda, 7-DOF, standard MuJoCo model.
- **Trajectory family**: *moving target* in the form of randomised quintic-Hermite waypoint splines (C² continuous, interpolated in cylindrical coordinates). The same policy is also evaluated on analytic shapes — circle and figure-8 — to demonstrate generalisation beyond the training family.
- **What's tracked**: position **and** orientation. Orientation is encoded as a 6-D rotation in observations and as a geodesic-angle reward.
- **Sources of uncertainty** — three, all simultaneously active during training:
  - **Observation noise**: Uniform on EE pose / velocity / orientation and on the joint state.
  - **Stochastic action delay**: every command is held in a buffer for 1–5 control steps.
  - **Domain randomization**: link inertia ±15 %, joint friction 0–0.2 N·m resampled on every episode reset.
- **RL component**: PPO with **asymmetric actor–critic** (RSL-RL inside mjlab): the actor sees the realistic noisy/delayed observation, the critic sees the clean privileged state.
- **Smoothness mechanism**: action is parametrised as a **delta-joint** position (zero output ≡ hold pose, no DC offset to learn); the reward adds 1st-order (action-rate) and 2nd-order (action-jerk) smoothness penalties.

## Results

Deterministic eval on the **circle (r = 15 cm, 1 cycle / 8 s)** and the **figure-8 (scale = 18 cm, 1 cycle / 8 s)**, with the full sim-to-real uncertainty stack active.

**Accuracy and smoothness** — RL policy under the action-delay stack:

| Trajectory | Position RMSE ↓ | EE jerk RMS ↓ | Orientation RMSE ↓ |
|---|---:|---:|---:|
| Circle   | **6.49 mm** | 19.82 m/s³ | 1.69° |
| Figure-8 | **6.26 mm** | 33.61 m/s³ | 1.68° |

Sub-cm position RMSE, sub-2° orientation RMSE, over the full 8 s rollout
of each deterministic trajectory.

**Robustness to noise / dynamics mismatch** — same checkpoint, varying
how much of the training-time uncertainty stack is active at eval:

| Uncertainty active                                | Circle pos RMSE | Figure-8 pos RMSE |
|---|---:|---:|
| Observation noise only (delay off, DR off)        | 7.99 mm | 9.01 mm |
| + action delay (1–5 control steps)                | **6.49 mm** | **6.26 mm** |
| + DR (link inertia ±15 %, joint friction 0–0.2)   | 6.66 mm | 7.21 mm |

Two observations worth flagging:

1. **The policy performs *best* in its trained distribution.** Position
   RMSE goes from 7.99 mm with no action delay to 6.49 mm with delay on
   — because the policy has learned to *anticipate* the delay (the
   lookahead window is exactly there for this), and removing it at test
   time produces a small overshoot. Robustness here is not "the policy
   tolerates noise"; it's "the policy is *tuned for* noise and
   gracefully handles its absence too."
2. **Dynamics randomisation is essentially free.** Adding DR on top of
   delay only degrades the circle RMSE by 0.17 mm and the figure-8 by
   0.95 mm — both well within the noise floor of a single 8 s rollout.

**Generalisation to the full training distribution** — the circle and
figure-8 above are simple, narrow-workspace cases. The training
distribution is much harder (cylindrical-spline trajectories spanning a
±0.5 m × ±0.4 m × [0.25, 0.50] m box, including waypoints behind the
base that require large joint-1 rotations). Rolling out **256 parallel
envs for one 12 s episode each** under the full uncertainty stack,
the distribution of per-env metrics is:

| Metric           |   p50 (median) |    p95 |    max |
|---|---:|---:|---:|
| Position RMSE    | **15.2 mm**    |  61 mm | 156 mm |
| EE jerk RMS      | 125 m/s³       | 237 m/s³ | —    |
| Orientation RMSE | **3.8°**       | 18.5°  | 72.0° |

The **median 15 mm** position RMSE on the wide training distribution is
consistent with the 6–7 mm we measured on the narrower, deterministic
eval trajectories — the policy generalises well to the typical case.
The **p95 / max tail** is dominated by a known failure mode: trajectories
that require sweeping the base joint ~180° around the robot's vertical
axis. The reward landscape locally favours a "detach-from-reference
then re-attach" strategy over the continuous joint-1 rotation, because
the saturating Gaussian position kernel makes large transient errors
bounded in cost while the continuous rotation accumulates action-rate
and jerk penalties throughout.

**Classical-control sanity check** — under the same delay-only stack,
an OSC controller (damped-least-squares Jacobian inverse with FF) reaches
11.66 mm / 13.64 mm position RMSE on circle / figure-8 (with 6.04 / 10.91
m/s³ jerk and 0.36° / 0.58° orientation RMSE). The RL policy trades some
smoothness for ~2× better position tracking; OSC's analytic Jacobian
buys it the orientation gap almost for free. Not the headline of this
submission — just a reference point.

### Reproduce the numbers

```bash
# RL policy — circle + figure-8
uv run python -m hac2k26.eval.run \
    --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_<iter>.pt \
    --output-dir eval_outputs/rl \
    --trajectories circle figure8 --disable-dr

# Classical OSC baseline — same trajectories
uv run python -m hac2k26.eval.run \
    --controller osc \
    --output-dir eval_outputs/osc \
    --trajectories circle figure8 --disable-dr

# Robustness ablation: re-run the RL command above with/without --disable-action-delay
# and --disable-dr to swap the active uncertainty stack.

# Training-distribution stats (256 parallel envs)
uv run python scripts/eval_random.py \
    --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_<iter>.pt \
    --num-envs 256
```

## Videos

The circles and figure-8 below are *out-of-distribution shapes*: the policy was only ever trained on the randomised quintic-spline references. It generalises because the reference signal in the observation is trajectory-shape-agnostic (target pos/vel/orient + phase + lookahead), so the policy treats any sufficiently smooth reference the same way.

### In-distribution quintic

What the policy actually trained on. Randomised waypoint trajectories in the cylindrical workspace, regenerated every 6 s.

<p align="center"><img src="media/training/play_traj.gif" alt="play_traj" width="640"></p>

```bash
uv run play Hac2k26-EETracking-Franka --checkpoint-file <path>
```

### Small circle (r = 0.20 m)

Well inside the trained workspace, smooth tracking.

<p align="center"><img src="media/circle/circle_02.gif" alt="circle_02" width="640"></p>

```bash
uv run python scripts/eval_play.py \
    --checkpoint <path> \
    --trajectory circle --radius 0.2 --n-cycles 3 --duration 12
```

### Large circle (r = 0.40 m)

Pushes to the **edge of the trained workspace**. The policy still
tracks, but localised jitter and a small "hop" appear at the point of
maximum reach — graceful degradation as the reference approaches the
boundary of the training distribution.

<p align="center"><img src="media/circle/circle_04.gif" alt="circle_04" width="640"></p>

```bash
uv run python scripts/eval_play.py \
    --checkpoint <path> \
    --trajectory circle --radius 0.4 --n-cycles 3 --duration 30
```

### Figure-8

Analytic lemniscate.

<p align="center"><img src="media/figure8/figure8.gif" alt="figure8" width="640"></p>

```bash
uv run python scripts/eval_play.py \
    --checkpoint <path> \
    --trajectory figure8 --scale 0.3 --n-cycles 3 --duration 25
```

> Reducing `--duration` while keeping `--n-cycles` fixed increases the
> reference angular velocity — useful to stress-test the policy at
> higher speeds without changing the trajectory shape.

## Design choices

### Observation space (asymmetric: actor 77-D, critic 77-D)

The actor and critic share the same set of terms; the actor sees *noisy*
values, the critic sees *clean* ground truth (privileged training signal).

| Group | Term | Dim | Noise (actor only) |
|---|---|---:|---|
| **Proprioception** | joint position (7-DOF, rel. to home) | 7 | ±0.03 rad |
|                    | joint velocity                       | 7 | ±0.1 rad/s |
|                    | last action                          | 7 | — |
| **EE state (world frame)** | position           | 3 | ±5 mm |
|                            | linear velocity    | 3 | ±5 cm/s |
|                            | angular velocity   | 3 | ±0.1 rad/s |
|                            | orientation (6-D rotation) | 6 | ±0.01 |
| **Reference signal** | target position           | 3 | — |
|                      | target velocity           | 3 | — |
|                      | target orientation (6-D)  | 6 | — |
|                      | phase φ ∈ [0, 1]          | 1 | — |
|                      | **lookahead window** (5 × `p_ref` at 50 ms spacing) | 15 | — |
| **Explicit tracking errors** | `p_ref − p_ee`            | 3 | ±5 mm |
|                              | `v_ref − v_ee`            | 3 | ±5 cm/s |
|                              | orientation error (rotvec)| 3 | ±0.01 |
|                              | `ω_ref − ω_ee`            | 3 | ±0.1 rad/s |
| **Singularity awareness** | manipulability `w(q) = √det(J Jᵀ)` | 1 | — |
| | **Total** | **77** | |

Three non-obvious choices in this layout:

- **Lookahead window + phase**: the policy gets explicit access to where
  the reference will be 50–250 ms into the future. This is what enables
  anticipation under action delay (the policy can aim ahead).
- **Noise on the error channels matches noise on the raw EE state**.
  Otherwise the actor could bypass the sim-to-real-style observation
  corruption by reading the (otherwise clean) tracking error directly.
  The critic still receives clean errors.
- **Reference signal is intentionally clean (no noise)**. The reference
  is a *command* generated by the trajectory planner, not a *measurement*
  read from the world — in real deployment it is also clean by
  construction (it's the setpoint the controller is asked to track).
  The realistic uncertainty is on what the robot *measures* about its
  own state, which is why noise is applied to proprioception and EE
  state but not to `traj_*`.

### Action space

| Property | Value |
|---|---|
| **Type**            | Δ joint position (additive, per-joint) |
| **Dimension**       | 7 (Franka arm joints) |
| **Range**           | [−0.1, +0.1] rad per joint per control step |
| **Control rate**    | 50 Hz (env step = 20 ms) |
| **Pipeline**        | `a_t → DelayBuffer(lag ~ U{1..5})  →  q_target = q + delayed(a_t)` |

Why delta over absolute joint positions: zero output ≡ hold pose, so the
policy doesn't have to learn a DC offset to "stay still" — smoothness is
built into the action parameterisation.

### Reward

```
r =  w_pos  · two-scale Gaussian on ||p_ee − p_ref||         (coarse α=30,  fine α=400)
   + w_ori  · two-scale Gaussian on geodesic orientation err (coarse β=2,   fine β=20)
   + w_vel  · v_ee · v̂_ref                                   (tangent alignment)
   − w_l2   · ||p_ee − p_ref||                                (always-on dense pull)
   − w_rate · ||a_t − a_{t-1}||²                              (1st-order smoothness)
   − w_jerk · ||a_t − 2·a_{t-1} + a_{t-2}||²                  (2nd-order smoothness)
   − w_sing · max(0, w_min − w(q))²                           (one-sided singularity barrier)
```

| Weight  | Value  | Role |
|---|---:|---|
| `w_pos`  |  3.0   | shaped position reward (peaky near target) |
| `w_ori`  |  1.5   | shaped orientation reward |
| `w_vel`  |  0.5   | bonus for moving along the reference tangent |
| `w_l2`   | 10.0   | unbounded penalty on `‖err‖`; the *primary* tracking signal |
| `w_rate` |  0.01  | 1st-order action smoothness |
| `w_jerk` |  0.015 | 2nd-order action smoothness (caught the most jitter) |
| `w_sing` | 50.0   | hinge penalty triggered when `w(q) < w_min = 0.04` |

Two non-obvious choices:

- **Two-scale Gaussian**: a single Gaussian kernel either has a wide
  basin (poor gradient at sub-cm precision) or a narrow basin (vanishing
  gradient when far). Combining a coarse kernel with a fine kernel gives
  a usable reward gradient at *both* scales.
- **The `w_l2 = 10` term is the load-bearing one for accuracy.** The
  Gaussian kernels saturate to zero when far from the target, so on
  their own they don't penalise large deviations. The unbounded L2 term
  ensures that "drift away from the reference" always costs more than
  "track imperfectly", which is what prevents the policy from learning
  the detach-and-reattach failure mode under action penalties.

### Trajectory representation

Waypoints are sampled in a Cartesian workspace box anchored to the robot
base, then **lifted to cylindrical (r, θ, z) for spline interpolation**.
The chain rule (`ẋ = ṙ cos θ − r θ̇ sin θ`, …) converts position and velocity
back to Cartesian at evaluation time. Two reasons to interpolate in
cylindrical coordinates rather than directly in Cartesian:

1. **Homotopy around the base.** A straight Cartesian segment between two
   waypoints on opposite sides of the robot will cut through the base. The
   cylindrical lift arcs the segment *around* the z-axis instead, which is
   the homotopy class the arm can actually realise. Δθ between consecutive
   waypoints is unwrapped via `atan2(sin Δθ, cos Δθ)` so segments always
   take the short angular path.
2. **A minimum radius `r_min`.** Waypoints whose 2-D radial distance to the
   base would be below `r_min` are pushed outward — this keeps the spline
   away from joint-1's wrist-shoulder singularity region.

**Spline math.** Each segment from waypoint `p_i` (at local time τ = 0)
to `p_{i+1}` (at τ = T) is a quintic polynomial in cylindrical coordinates:

```
p(τ) = c₀ + c₁ τ + c₂ τ² + c₃ τ³ + c₄ τ⁴ + c₅ τ⁵
```

The six boundary conditions are `(p_i, v_i, a=0)` at τ = 0 and
`(p_{i+1}, v_{i+1}, a=0)` at τ = T — six BCs match six coefficients. Solving
gives the closed form used at every reset (with `Δp = p_{i+1} − p_i − v_i T`
and `Δv = v_{i+1} − v_i`):

```
c₀ = p_i                  c₃ = ( 10 Δp − 4 T Δv) / T³
c₁ = v_i                  c₄ = (−15 Δp + 7 T Δv) / T⁴
c₂ = 0                    c₅ = (  6 Δp − 3 T Δv) / T⁵
```

Forcing `a = 0` at every waypoint guarantees `C²` continuity at segment
boundaries (no acceleration discontinuities) and clamps the boundary jerk
— exactly what we want a robot tracking the reference to follow without
fighting acceleration jumps.

**Interior-waypoint velocities** (the `v_i` for `0 < i < K-1`) come from a
**centripetal-Catmull–Rom-style tangent**:

```
v_i = α · (p_{i+1} − p_{i−1}) / (2 T)
```

The endpoints stay at rest (`v_0 = v_{K-1} = 0`). The through-speed
coefficient α is sampled per trajectory in `[α_min · vel_scale, vel_scale]`,
so the curriculum can interpolate from rest-to-rest (α = 0, stop at every
waypoint) to fully continuous flow (α near 1) without a discontinuity in
trajectory *style* — the spline form is identical, only the interior
tangent magnitude changes.

**Curriculum.** A linear schedule on `vel_scale` (0 → 1.2) and on the
per-trajectory waypoint-sampling radius runs over the first ~50 k env
steps, so the policy first learns to *reach* a single waypoint at rest,
then to *flow* through a chain of them at speed.

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

- Trained for ~10 k PPO iterations (~3 hours on a single RTX 4070)
  with 4096 parallel environments under the full uncertainty stack.
- All hyper-parameters in `src/hac2k26/ee_tracking/config/franka/rl_cfg.py`.
- Trajectory hyper-parameters in `src/hac2k26/ee_tracking/tracking_env_cfg.py`.


