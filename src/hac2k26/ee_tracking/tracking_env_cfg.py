"""EE tracking task configuration.

Base factory function for end-effector trajectory tracking.
Robot-specific configurations call the factory and customize as needed.
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


def _proprio_actor_obs() -> dict[str, ObservationTermCfg]:
    return {
        "ee_pos_w": ObservationTermCfg(
            func=mdp.observations.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "ee_lin_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
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
    }


def _proprio_critic_obs() -> dict[str, ObservationTermCfg]:
    return {
        "ee_pos_w": ObservationTermCfg(
            func=mdp.observations.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
        ),
        "ee_lin_vel_w": ObservationTermCfg(
            func=mdp.observations.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
        ),
        "ee_rot6d_w": ObservationTermCfg(
            func=mdp.observations.ee_rot6d_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("hand",))},
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def make_ee_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base EE tracking task configuration."""

    ##
    # Observations
    ##

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

    ##
    # Actions (placeholder — delta joint positions)
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": RelativeJointPositionActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-7]",),
            scale=0.5,
        )
    }

    ##
    # Rewards (placeholder)
    ##

    rewards: dict[str, RewardTermCfg] = {
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    }

    ##
    # Terminations
    ##

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
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
        episode_length_s=10.0,
    )
