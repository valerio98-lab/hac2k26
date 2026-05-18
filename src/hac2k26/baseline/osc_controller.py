from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.differential_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)
from mjlab.utils.lab_api.math import quat_box_plus

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class OSCController:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        kp_pos: float = 12.0,
        kp_ori: float = 4.0,
        damping: float = 0.05,
        max_dq: float = 0.1,
        action_scale: float = 0.1,
        use_velocity_feedforward: bool = True,
        command_name: str = "trajectory",
    ) -> None:
        self._env = env
        self._action_scale = float(action_scale)
        self._use_ff = bool(use_velocity_feedforward)
        self._command_name = command_name

        self._dik = DifferentialIKAction(
            DifferentialIKActionCfg(
                entity_name="robot",
                actuator_names=("joint[1-7]",),
                frame_name="hand",
                use_relative_mode=False,
                position_weight=kp_pos,
                orientation_weight=kp_ori,
                damping=damping,
                max_dq=max_dq,
            ),
            env,
        )

    def __call__(self, obs=None) -> torch.Tensor:
        del obs
        cmd = self._env.command_manager.get_command(self._command_name)
        p_ref = cmd[:, 0:3]
        v_ref = cmd[:, 3:6]
        q_ref = cmd[:, 7:11]

        if self._use_ff:
            dt = self._env.step_dt
            omega_ref = cmd[:, 11:14]
            p_target = p_ref + v_ref * dt
            q_target = quat_box_plus(q_ref, omega_ref * dt)
        else:
            p_target = p_ref
            q_target = q_ref

        self._dik.process_actions(torch.cat([p_target, q_target], dim=-1))
        return self._dik.compute_dq() / self._action_scale
