from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hac2k26.ee_tracking import mdp

_HAND = SceneEntityCfg("robot", body_names=("hand",))
_ARM = SceneEntityCfg("robot", joint_names=("joint[1-7]",))


def _proprio_actor_obs() -> dict[str, ObservationTermCfg]:
    return {
        "ee_pos_w": ObservationTermCfg(
            func=mdp.observations.ee_pos_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "ee_lin_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_lin_vel_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "ee_ang_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_ang_vel_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"biased": True, "asset_cfg": _ARM},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": _ARM},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        # Reference signal.
        "traj_pos_ref": ObservationTermCfg(func=mdp.observations.traj_pos_ref),
        "traj_vel_ref": ObservationTermCfg(func=mdp.observations.traj_vel_ref),
        "traj_rot6d_ref": ObservationTermCfg(func=mdp.observations.traj_rot6d_ref),
        "traj_phase": ObservationTermCfg(func=mdp.observations.traj_phase),
        "traj_lookahead": ObservationTermCfg(func=mdp.observations.traj_lookahead),
        # Explicit tracking errors — the biggest single speed-up for tracking
        # tasks; saves the network from learning the subtraction.
        #
        # These are computed from the *ground-truth* EE state inside the
        # observation function (asset.data.body_link_*), so they would
        # bypass the noise applied to ``ee_pos_w`` / ``ee_lin_vel_w`` /
        # ``ee_rot6d_w`` / ``ee_ang_vel_w`` if left clean. We inject the
        # SAME magnitude of noise here so the actor cannot dodge sensor
        # noise by reading the (clean) error channel — proper sim-to-real
        # discipline. The critic still sees clean errors (privileged).
        "pos_error_w": ObservationTermCfg(
            func=mdp.observations.pos_error_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "lin_vel_error_w": ObservationTermCfg(
            func=mdp.observations.lin_vel_error_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "rot_error_rotvec": ObservationTermCfg(
            func=mdp.observations.rot_error_rotvec,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ang_vel_error_w": ObservationTermCfg(
            func=mdp.observations.ang_vel_error_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        # Singularity awareness: linear manipulability w(q).
        "manipulability": ObservationTermCfg(
            func=mdp.kinematics.manipulability,
            params={"asset_cfg": _HAND, "joint_pattern": "joint[1-7]"},
        ),
    }


def _proprio_critic_obs() -> dict[str, ObservationTermCfg]:
    return {
        "ee_pos_w": ObservationTermCfg(
            func=mdp.observations.ee_pos_w, params={"asset_cfg": _HAND}
        ),
        "ee_lin_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_lin_vel_w, params={"asset_cfg": _HAND}
        ),
        "ee_ang_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_ang_vel_w, params={"asset_cfg": _HAND}
        ),
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w, params={"asset_cfg": _HAND}
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": _ARM}
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": _ARM}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "traj_pos_ref": ObservationTermCfg(func=mdp.observations.traj_pos_ref),
        "traj_vel_ref": ObservationTermCfg(func=mdp.observations.traj_vel_ref),
        "traj_rot6d_ref": ObservationTermCfg(func=mdp.observations.traj_rot6d_ref),
        "traj_phase": ObservationTermCfg(func=mdp.observations.traj_phase),
        "traj_lookahead": ObservationTermCfg(func=mdp.observations.traj_lookahead),
        "pos_error_w": ObservationTermCfg(
            func=mdp.observations.pos_error_w, params={"asset_cfg": _HAND}
        ),
        "lin_vel_error_w": ObservationTermCfg(
            func=mdp.observations.lin_vel_error_w, params={"asset_cfg": _HAND}
        ),
        "rot_error_rotvec": ObservationTermCfg(
            func=mdp.observations.rot_error_rotvec, params={"asset_cfg": _HAND}
        ),
        "ang_vel_error_w": ObservationTermCfg(
            func=mdp.observations.ang_vel_error_w, params={"asset_cfg": _HAND}
        ),
        "manipulability": ObservationTermCfg(
            func=mdp.kinematics.manipulability,
            params={"asset_cfg": _HAND, "joint_pattern": "joint[1-7]"},
        ),
    }


def _local_rewards() -> dict[str, RewardTermCfg]:
    """Reward weights paired with the local cartesian sampling regime."""
    return {
        "tracking_pos": RewardTermCfg(
            func=mdp.rewards.pos_reward,
            weight=3.0,
            params={
                "asset_cfg": _HAND,
                "alpha_coarse": 30.0,
                "alpha_fine": 400.0,
                "fine_weight": 0.5,
            },
        ),
        "tracking_pos_l2": RewardTermCfg(
            func=mdp.rewards.pos_error_l2_penalty,
            weight=-1.0,
            params={"asset_cfg": _HAND},
        ),
        "tracking_vel": RewardTermCfg(
            func=mdp.rewards.velocity_alignment_reward,
            weight=0.5,
            params={"asset_cfg": _HAND},
        ),
        "tracking_ori": RewardTermCfg(
            func=mdp.rewards.orientation_reward,
            weight=1.0,
            params={
                "asset_cfg": _HAND,
                "beta_coarse": 2.0,
                "beta_fine": 20.0,
                "fine_weight": 0.5,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "action_jerk_l2": RewardTermCfg(func=mdp.rewards.action_jerk_l2, weight=-0.015),
        "singularity_penalty": RewardTermCfg(
            func=mdp.kinematics.singularity_penalty,
            weight=-50.0,
            params={
                "asset_cfg": _HAND,
                "joint_pattern": "joint[1-7]",
                "w_min": 0.04,
            },
        ),
    }


def _global_rewards() -> dict[str, RewardTermCfg]:
    """Reward weights paired with the global cylindrical sampling regime."""
    return {
        # Two-scale kernel: gradient both far (~tens of cm) and very close
        # (sub-cm).
        "tracking_pos": RewardTermCfg(
            func=mdp.rewards.pos_reward,
            weight=3.2,
            params={
                "asset_cfg": _HAND,
                "alpha_coarse": 30.0,
                "alpha_fine": 400.0,
                "fine_weight": 0.5,
            },
        ),
        # Stronger L2 floor than local mode: at global angular sweeps the
        # robot can lose the reference; the quadratic pull is what brings
        # it back without giving up.
        "tracking_pos_l2": RewardTermCfg(
            func=mdp.rewards.pos_error_l2_penalty,
            weight=-2.5,
            params={"asset_cfg": _HAND},
        ),
        "tracking_vel": RewardTermCfg(
            func=mdp.rewards.velocity_alignment_reward,
            weight=0.08,
            params={"asset_cfg": _HAND},
        ),
        "tracking_ori": RewardTermCfg(
            func=mdp.rewards.orientation_reward,
            weight=2.0,
            params={
                "asset_cfg": _HAND,
                "beta_coarse": 2.0,
                "beta_fine": 20.0,
                "fine_weight": 0.5,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03),
        "action_jerk_l2": RewardTermCfg(func=mdp.rewards.action_jerk_l2, weight=-0.20),
        # Singularity awareness — pushed harder in global mode where the arm
        # is more often near reconfiguration boundaries.
        "singularity_penalty": RewardTermCfg(
            func=mdp.kinematics.singularity_penalty,
            weight=-200.0,
            params={
                "asset_cfg": _HAND,
                "joint_pattern": "joint[1-7]",
                "w_min": 0.04,
            },
        ),
    }


def _local_trajectory_cfg() -> "mdp.commands.EETrackingCommandCfg":
    return mdp.commands.EETrackingCommandCfg(
        sampling_mode="local",
        num_waypoints=5,
        segment_duration=1.5,
        debug_vis=True,
        workspace_low=(0.20, -0.40, 0.15),
        workspace_high=(0.70, 0.40, 0.80),
        lookahead_steps=5,
        lookahead_dt=0.05,
        anchor_first_waypoint=True,
        radius_start=0.15,
        radius_end=0.45,
        ori_radius_start=0.20,
        ori_radius_end=0.80,
        vel_scale_start=0.0,
        vel_scale_end=0.8,
        curriculum_steps=100_000,
        asset_name="robot",
        body_name="hand",
    )


def _global_trajectory_cfg() -> "mdp.commands.EETrackingCommandCfg":
    # K=7, Δθ_max=1.2 sets the per-segment peak rate to 1.875·Δθ/T_seg
    # = 1.88 rad/s — 86% of joint1's 2.175 rad/s velocity cap. The
    # cumulative unwrapped theta is hard-clamped at ``theta_max_rad``
    # (default 2.5 rad, inside joint1's 2.608 rad soft position limit)
    # so the sweep saturates at the joint limit instead of demanding
    # physically infeasible angular reach. With K-1=6 segments the
    # cumulative could otherwise reach 7.2 rad — well past joint1's
    # bound — and produce the "skipping segments" pathology observed
    # with the previous (K=5, Δθ_max=2.0) sweep.
    return mdp.commands.EETrackingCommandCfg(
        sampling_mode="global",
        num_waypoints=7,
        segment_duration=1.2,
        debug_vis=True,
        lookahead_steps=5,
        lookahead_dt=0.05,
        anchor_first_waypoint=True,
        r_min=0.35,
        r_max=0.65,
        z_min=0.20,
        z_max=0.75,
        dtheta_max_start=0.3,
        dtheta_max_end=1.2,
        theta_max_rad=2.5,
        ori_radius_start=0.20,
        ori_radius_end=0.80,
        vel_scale_start=0.0,
        vel_scale_end=0.7,
        curriculum_steps=100_000,
        asset_name="robot",
        body_name="hand",
    )


def make_ee_tracking_env_cfg(
    sampling_mode: Literal["local", "global"] = "global",
) -> ManagerBasedRlEnvCfg:
    """Create base EE tracking task configuration.

    Args:
        sampling_mode: ``"local"`` for cartesian quintic in a cube around the
            EE; ``"global"`` for cylindrical quintic with angular sweeps.
            Reward weights and trajectory params are paired with the chosen
            mode.
    """

    actor_terms = _proprio_actor_obs()
    critic_terms = _proprio_critic_obs()

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": mdp.actions.DelayedRelativeJointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-7]",),
            # Delta-joint scale. With kp~4500 and force_limit=87 Nm, the
            # actuator saturates above |delta| ≈ 0.02 rad/substep. Cap the
            # per-step delta to ~0.1 rad so the action distribution is
            # mostly inside the linear regime of the PD controller.
            scale=0.1,
            min_lag=1,
            max_lag=5,
        )
    }

    if sampling_mode == "local":
        rewards = _local_rewards()
        trajectory_cfg = _local_trajectory_cfg()
    elif sampling_mode == "global":
        rewards = _global_rewards()
        trajectory_cfg = _global_trajectory_cfg()
    else:
        raise ValueError(
            f"sampling_mode must be 'local' or 'global', got {sampling_mode!r}"
        )

    commands = {"trajectory": trajectory_cfg}

    _DR_BODIES = SceneEntityCfg(
        "robot",
        body_names=(
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "left_finger",
            "right_finger",
        ),
    )
    _DR_ARM_JOINTS = SceneEntityCfg("robot", joint_names=("joint[1-7]",))
    events = {
        "reset_robot": EventTermCfg(func=mdp.events.reset_home, mode="reset"),
        # Domain randomization is fired on every episode reset so each
        # rollout sees a freshly sampled dynamics, instead of one fixed
        # draw per parallel worker (mode="startup"). The event manager
        # batches recomputes across all reset terms, so the heavy
        # ``set_const`` recompute triggered by ``pseudo_inertia`` runs
        # at most once per step regardless of how many envs reset.
        "dr_pseudo_inertia": EventTermCfg(
            func=mdp.dr.pseudo_inertia,
            mode="reset",
            params={"alpha_range": (-0.07, 0.07), "asset_cfg": _DR_BODIES},
        ),
        "dr_joint_friction": EventTermCfg(
            func=mdp.dr.joint_friction,
            mode="reset",
            params={
                "ranges": (0.0, 0.2),
                "operation": "abs",
                "asset_cfg": _DR_ARM_JOINTS,
            },
        ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    # Episode length covers 2 full trajectories.
    episode_length_s = (
        2.0 * trajectory_cfg.num_waypoints * trajectory_cfg.segment_duration
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="link0",
            distance=1.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=episode_length_s,
    )
