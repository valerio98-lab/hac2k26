"""Action terms for the EE tracking task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import (
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from mjlab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Relative joint position control with stochastic action delay.

    A DelayBuffer holds the last ``max_lag + 1`` raw policy outputs per env
    and serves one delayed by a lag sampled uniformly in
    ``[min_lag, max_lag]`` at each control step..
    """

    min_lag: int = 1
    """Minimum action delay in control steps."""

    max_lag: int = 5
    """Maximum action delay in control steps."""

    def build(self, env: ManagerBasedRlEnv) -> DelayedRelativeJointPositionAction:
        return DelayedRelativeJointPositionAction(self, env)


class DelayedRelativeJointPositionAction(RelativeJointPositionAction):
    """RelativeJointPositionAction wrapped in a stochastic delay buffer."""

    def __init__(
        self,
        cfg: DelayedRelativeJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ):
        super().__init__(cfg=cfg, env=env)
        self._delay_buf = DelayBuffer(
            min_lag=cfg.min_lag,
            max_lag=cfg.max_lag,
            batch_size=env.num_envs,
            device=str(env.device),
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._delay_buf.append(actions)
        delayed = self._delay_buf.compute()
        super().process_actions(delayed)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._delay_buf.reset(batch_ids=env_ids)
