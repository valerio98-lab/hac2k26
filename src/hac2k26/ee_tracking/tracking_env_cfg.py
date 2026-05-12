"""EE tracking task configuration.

Base factory function for end-effector trajectory tracking.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hac2k26.ee_tracking import mdp

_HAND = SceneEntityCfg("robot", body_names=("hand",))


def _proprio_actor_obs() -> dict[str, ObservationTermCfg]:
    return {
        # Robot state (noisy — what real sensors would give).
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
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w,
            params={"asset_cfg": _HAND},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"biased": True},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "traj_pos_ref": ObservationTermCfg(func=mdp.observations.traj_pos_ref),
        "traj_vel_ref": ObservationTermCfg(func=mdp.observations.traj_vel_ref),
        "traj_phase": ObservationTermCfg(func=mdp.observations.traj_phase),
        "traj_lookahead": ObservationTermCfg(func=mdp.observations.traj_lookahead),
    }


def _proprio_critic_obs() -> dict[str, ObservationTermCfg]:
    return {
        "ee_pos_w": ObservationTermCfg(
            func=mdp.observations.ee_pos_w, params={"asset_cfg": _HAND}
        ),
        "ee_lin_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_lin_vel_w, params={"asset_cfg": _HAND}
        ),
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w, params={"asset_cfg": _HAND}
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "traj_pos_ref": ObservationTermCfg(func=mdp.observations.traj_pos_ref),
        "traj_vel_ref": ObservationTermCfg(func=mdp.observations.traj_vel_ref),
        "traj_phase": ObservationTermCfg(func=mdp.observations.traj_phase),
        "traj_lookahead": ObservationTermCfg(func=mdp.observations.traj_lookahead),
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
        "joint_pos": RelativeJointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-7]",),
            scale=0.5,
        )
    }

    rewards: dict[str, RewardTermCfg] = {
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    }

    commands = {
        "trajectory": mdp.commands.EETrackingCommandCfg(
            num_waypoints=5,
            segment_duration=1.5,
            workspace_low=(0.30, -0.30, 0.20),
            workspace_high=(0.60, 0.30, 0.60),
            lookahead_steps=5,
            lookahead_dt=0.05,
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
        events={},
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
            njmax=200,
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
