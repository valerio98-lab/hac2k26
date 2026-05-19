# EE Tracking on a Franka Panda

A reinforcement-learning policy that drives a 7-DOF Franka Panda to track arbitrary 3D end-effector trajectories (position and orientation) while staying robust to sensor noise, action delay, and dynamics mismatch.

Built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + Isaac-Lab-style manager API) and RSL-RL PPO.

<p align="center"><img src="media/training/play_traj.gif" width="640"></p>

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/valerio98-lab/hac2k26.git && cd hac2k26
uv sync

# train
uv run train Hac2k26-EETracking-Franka --num_envs 4096

# play a trained checkpoint
uv run play Hac2k26-EETracking-Franka --checkpoint-file <path>
```

Training takes about three hours on a single RTX 4070.

## How it works

**State.** The actor sees 77 numbers per step: joint positions and velocities, end-effector pose and velocity in world frame, the current reference (target pose, target velocity, a phase variable, and a 5-step lookahead window at 50 ms spacing), explicit tracking errors so the policy doesn't have to subtract them itself, and a scalar manipulability measure √det(J Jᵀ) for singularity awareness. The critic sees the same terms with the noise stripped.

**Action.** delta-joint position at 50 Hz, clipped to ±0.1 rad per joint per step. Delta-control is a standard choice in humanoid RL (DeepMimic-style residual actions on top of a reference, and most modern locomotion/loco-manipulation work): zero output means "hold pose", so smoothness is built into the parameterization.

**Reward.** Sum of:

- **Position tracking** — two-scale Gaussian on ‖p_ref − p_ee‖, with a coarse and a fine kernel combined to avoid the *cliff effect*: a single Gaussian either has a wide basin with no sub-cm gradient, or a narrow basin that gives zero gradient when far. An unbounded L2 penalty sits on top so the policy can't drift when both kernels saturate.
- **Orientation tracking** — two-scale Gaussian on the geodesic angle between target and current EE quaternion (same coarse/fine reasoning as above).
- **Velocity alignment** — `v_ee · v̂_ref`, a small bonus for moving along the reference tangent.
- **Smoothness** — penalties on action rate (first derivative) and action jerk (second derivative). The jerk term is the one that kills the high-frequency jitter visible at play time.
- **Singularity barrier** — one-sided hinge on the manipulability deficit `max(0, w_min − w(q))²`, active only when the arm gets close to a singular configuration.

**Trajectory.** Random splines through six waypoints, regenerated every six seconds. Each segment is a quintic polynomial

```
p(τ) = c₀ + c₁ τ + c₂ τ² + c₃ τ³ + c₄ τ⁴ + c₅ τ⁵
```

with six coefficients matched against six boundary conditions: position, velocity, and zero acceleration at both endpoints. This gives C² continuity at every waypoint, and bounds the jerk by construction (no infinite spikes at the joins).

Waypoints are sampled in a workspace box, but the spline itself is fit in cylindrical (r, θ, z) coordinates and mapped back to Cartesian. A straight Cartesian segment between waypoints on opposite sides of the base would cut through the robot, while the cylindrical lift arcs around it. A minimum radius keeps the spline away from the shoulder-wrist singularity region.

Interior waypoint velocities come from a centripetal Catmull-Rom tangent scaled by a curriculum coefficient that ramps from rest-to-rest to fully continuous flow.

Orientation is generated independently. Random target quaternions at each waypoint, interpolated with the same quintic smoothstep applied to the axis-angle log (a SLERP equivalent with zero angular velocity at every waypoint).

**Uncertainty.** Three sources, all active during training:


- **Observation noise** — uniform, on every measurement-like channel (proprio, EE state, tracking errors). Trajectory reference signals stay clean: they are commands input into the policy, not sensor readings.
- **Action delay** — each command held in a buffer for a stochastic 1–5 control steps before reaching the simulator.
- **Domain randomization** — link inertia (±15%) and joint friction (0–0.2 N·m), resampled at every episode reset.

**RL.** PPO with an asymmetric actor-critic. The actor trains on the noisy and delayed observation, the critic sees the clean ground truth. The lookahead window in the observation is what makes anticipation under delay possible: the policy can aim ahead.

## Results

Position RMSE in mm, orientation RMSE in degrees, jerk RMS in m/s³. Full uncertainty stack unless stated. `p50` and `p95` are the median and the 95th percentile across 256 parallel envs.

| | Pos RMSE | Ori RMSE | Jerk RMS |
|---|---:|---:|---:|
| Training distribution, p50 (256 envs)   | 15.2 | 3.8  | 125  |
| Training distribution, p95              | 61   | 18.5 | 237  |
| Circle, r = 15 cm (OOD)                 | **6.5**  | 1.7  | 19.8 |
| Figure-8, scale = 18 cm (OOD)           | **6.3**  | 1.7  | 33.6 |

Circle and figure-8 are out-of-distribution: the policy only ever saw random quintic splines during training. It generalizes because the reference signal is shape-agnostic (target pose, velocity, phase, lookahead), so any sufficiently smooth reference looks the same to the policy.

**Robustness ablation** (same checkpoint, varying uncertainty at evaluation):

| | Circle | Figure-8 |
|---|---:|---:|
| Noise only                              | 8.0 mm | 9.0 mm |
| Noise + action delay                    | **6.5 mm** | **6.3 mm** |
| Noise + delay + DR                      | 6.7 mm | 7.2 mm |

The policy tracks *best* with the training-time delay active. The reason is that the policy has learned to use the lookahead window to compensate for the delay — effectively aiming a few control steps ahead so that, by the time the action lands, the EE is on the reference. Removing the delay at evaluation breaks that compensation: the action arrives instantly while the policy is still aiming ahead, producing a small but consistent overshoot. Adding domain randomization on top of the delay costs at most ~1 mm.

## Videos

<p align="center">
  <img src="media/circle/circle_02.gif" width="380">
  <img src="media/figure8/figure8.gif" width="380">
</p>

Small circle (r = 20 cm) and figure-8 (scale = 30 cm). A larger circle (r = 40 cm) reaches the edge of the trained workspace; the policy still tracks, with localized jitter and a small "hop" at maximum reach — graceful degradation as the reference approaches the workspace boundary.

<p align="center"><img src="media/circle/circle_04.gif" width="640"></p>

## Reproducing

```bash
# Deterministic circle + figure-8
uv run python -m hac2k26.eval.run \
    --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_<iter>.pt \
    --output-dir eval_outputs/rl \
    --trajectories circle figure8 --disable-dr

# Training-distribution stats across 256 parallel envs
uv run python scripts/eval_random.py \
    --checkpoint logs/rsl_rl/franka_ee_tracking/<run>/model_<iter>.pt \
    --num-envs 256
```

Add or remove `--disable-action-delay` and `--disable-dr` to reproduce the robustness ablation.

## Code layout

```
src/hac2k26/
  ee_tracking/
    tracking_env_cfg.py       env factory: obs, rewards, events, command
    config/franka/            task registration + PPO hyperparameters
    mdp/
      commands.py             quintic-spline reference generator
      actions.py              Δ-joint action with delay buffer
      rewards.py              tracking + smoothness + singularity
      observations.py         actor/critic observation terms
      kinematics.py           manipulability via warp Jacobian
      events.py               reset + domain randomization
  eval/run.py                 deterministic evaluation entry point
scripts/                      training-distribution eval, plotting, play
```

PPO hyperparameters in `src/hac2k26/ee_tracking/config/franka/rl_cfg.py`. Environment and trajectory parameters in `src/hac2k26/ee_tracking/tracking_env_cfg.py`.
