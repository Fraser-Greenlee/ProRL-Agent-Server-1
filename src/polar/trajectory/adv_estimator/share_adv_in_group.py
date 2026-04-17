"""Communal advantage estimator for multi-agent trajectories.

Every trace inside a trajectory shares one outcome reward — this is the
communal case where subagents / parallel agents inside one agent session
are graded by a single outcome reward and share credit equally (一个
OpenCode 里所有 agent traces 吃大锅饭). The shared reward is the
trajectory outcome; outcomes are GRPO-standardized across the group,
and the resulting advantage is written back to every trace in the
trajectory.

Rewards within one trajectory must be consistent: any non-``None`` trace
rewards must all be identical. This accepts both explicit
``[R, R, …, R]`` (all traces carry the shared reward) and the default
gateway output ``[None, …, None, R]`` (outcome reward landed only on the
last trace). Conflicting per-trace rewards raise ``ValueError`` — use
``hierarchical_group`` when you really want per-trace credit assignment.
"""

from __future__ import annotations

from typing import Any

from polar.trajectory.adv_estimator._stats import standardize
from polar.trajectory.adv_estimator.base import BaseAdvantageEstimator
from polar.trajectory.models import Trajectory


class ShareAdvInGroupAdvantageEstimator(BaseAdvantageEstimator):
    """All traces in a trajectory share one advantage derived from the group.

    Reduces to standard GRPO at the trajectory level — each trajectory
    contributes a single outcome to the group statistic regardless of how
    many traces it contains.
    """

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)

    def estimate(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        if not trajectories:
            return trajectories

        outcomes = [self._shared_reward(traj) for traj in trajectories]
        advantages = standardize([o if o is not None else 0.0 for o in outcomes])

        updated: list[Trajectory] = []
        for traj, advantage in zip(trajectories, advantages):
            new_traces = [
                trace.model_copy(update={"advantage": advantage})
                for trace in traj.traces
            ]
            updated.append(traj.model_copy(update={"traces": new_traces}))

        return updated

    def _shared_reward(self, traj: Trajectory) -> float | None:
        rewards = [trace.reward for trace in traj.traces]
        non_none = [r for r in rewards if r is not None]
        if not non_none:
            return None
        first = non_none[0]
        for reward in non_none[1:]:
            if reward != first:
                raise ValueError(
                    "share_adv_in_group requires a single shared reward per "
                    f"trajectory; got conflicting trace rewards {rewards!r}"
                )
        return first
