"""EE tracking task configuration.

Base factory function for end-effector trajectory tracking.
"""

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
_HAND_ARM = SceneEntityCfg("robot", body_names=("hand",), joint_names=("joint[1-7]",))


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


def make_ee_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base EE tracking task configuration."""

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

    rewards: dict[str, RewardTermCfg] = {
        # Two-scale kernel: gradient both far (~tens of cm) and very close
        # (sub-cm). Replaces the prior alpha=10 kernel that saturated as soon
        # as the EE was vaguely near the reference.
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
        # Bumped from -0.005 to -0.015 to attack the residual jitter still
        # visible at play time on figure-8 cusps and singular configurations.
        "action_jerk_l2": RewardTermCfg(
            func=mdp.rewards.action_jerk_l2, weight=-0.1
        ),  # -0.015
        # Singularity awareness. Hinge: zero contribution when w(q) >= w_min
        # (~28% of the home-pose manipulability of 0.143), quadratic deficit
        # otherwise. Weight is small so the policy is nudged away from
        # singular regions without sacrificing tracking precision.
        "singularity_penalty": RewardTermCfg(
            func=mdp.kinematics.singularity_penalty,
            # ``(w_min - w)^2`` saturates at ``w_min^2 = 1.6e-3`` (when
            # w -> 0). Weight is chosen so the worst-case contribution is
            # ~5% of the tracking_pos signal, enough to push away from
            # singular regions but not enough to dominate trajectory
            # following.
            weight=-300.0,  # -50
            params={
                "asset_cfg": _HAND,
                "joint_pattern": "joint[1-7]",
                "w_min": 0.04,
            },
        ),
    }

    commands = {
        "trajectory": mdp.commands.EETrackingCommandCfg(
            num_waypoints=5,
            segment_duration=1.0,  # 1.5
            debug_vis=True,
            workspace_low=(0.20, -0.40, 0.10),
            workspace_high=(0.70, 0.40, 0.80),
            lookahead_steps=5,
            lookahead_dt=0.05,
            anchor_first_waypoint=True,
            r_min=0.25,
            r_max=0.65,
            z_min=0.20,
            z_max=0.65,
            dtheta_max_start=0.5,
            dtheta_max_end=3.14,
            ori_radius_start=0.20,
            ori_radius_end=0.80,
            # Variable waypoint velocities — rest-to-rest at the very
            # start of training, ramp to a non-trivial through-speed by the
            # end of the curriculum window. With centripetal Catmull-Rom
            # tangents and radius=0.35m, T_seg=1.5s, max |v_waypoint| stays
            # below ~0.2 m/s — well within Franka joint-velocity limits.
            vel_scale_start=0.8,  # 0.0
            vel_scale_end=1.2,  # 0.8
            curriculum_steps=30_000,
            asset_name="robot",
            body_name="hand",
        ),
    }
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
        "dr_pseudo_inertia": EventTermCfg(
            func=mdp.dr.pseudo_inertia,
            mode="startup",
            params={"alpha_range": (-0.07, 0.07), "asset_cfg": _DR_BODIES},
        ),
        "dr_joint_friction": EventTermCfg(
            func=mdp.dr.joint_friction,
            mode="startup",
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
        # Long enough to cover 2 full (5-waypoint × 1.5s) trajectories.
        episode_length_s=12.0,
    )
