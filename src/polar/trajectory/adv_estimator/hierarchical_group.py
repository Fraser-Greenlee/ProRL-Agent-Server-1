"""Hierarchical group advantage estimator for multi-trace trajectories.

Decomposes advantage into two levels:

    A(trace j in trajectory i) = (1/K_i) * [A_between(i) + beta * A_within(i, j)]

- **Between-trajectory** (GRPO-style): compares trajectory-level aggregate
  outcomes across the group.
- **Within-trajectory**: compares individual trace rewards against siblings
  in the same trajectory, giving per-trace credit assignment.
- **1/K_i weighting**: normalizes so each trajectory contributes equally to
  gradient regardless of its trace count.

All traces in ``Trajectory.traces`` are assumed to be training targets
(builders own trace curation).

Reduces to standard GRPO when every trajectory has exactly one trace.
"""

from __future__ import annotations

import statistics
from typing import Any

from polar.trajectory.adv_estimator.base import BaseAdvantageEstimator
from polar.trajectory.models import Trajectory

_EPS = 1e-8


class HierarchicalGroupAdvantageEstimator(BaseAdvantageEstimator):
    """Two-level advantage: between-trajectory GRPO + within-trajectory credit.

    Config keys:
        beta:      Weight for within-trajectory advantage (default 0.5).
                   0 = pure GRPO, 1 = equal weight between/within.
        aggregate: How to summarise trace rewards into a trajectory outcome.
                   "mean" (default) or "max".
    """

    def __init__(self, *, beta: float = 0.5, aggregate: str = "mean", **config: Any) -> None:
        super().__init__(**config)
        self.beta = beta
        if aggregate not in ("mean", "max"):
            raise ValueError(f"aggregate must be 'mean' or 'max', got {aggregate!r}")
        self.aggregate = aggregate

    def estimate(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        if not trajectories:
            return trajectories

        outcomes = [self._trajectory_outcome(traj) for traj in trajectories]
        between_advs = _standardize(outcomes)

        updated: list[Trajectory] = []
        for traj, between_adv in zip(trajectories, between_advs):
            trace_rewards = [
                t.reward if t.reward is not None else 0.0
                for t in traj.traces
            ]
            within_advs = _standardize(trace_rewards)
            k = len(traj.traces) or 1
            new_traces = []
            for trace, within_adv in zip(traj.traces, within_advs):
                advantage = (between_adv + self.beta * within_adv) / k
                new_traces.append(trace.model_copy(update={"advantage": advantage}))
            updated.append(traj.model_copy(update={"traces": new_traces}))

        return updated

    def _trajectory_outcome(self, traj: Trajectory) -> float:
        rewards = [t.reward for t in traj.traces if t.reward is not None]
        if not rewards:
            return 0.0
        if self.aggregate == "max":
            return max(rewards)
        return statistics.mean(rewards)


def _standardize(values: list[float]) -> list[float]:
    """(x - mean) / std, returns zeros when std ~ 0 or len <= 1."""
    n = len(values)
    if n <= 1:
        return [0.0] * n
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values)
    if sigma < _EPS:
        return [0.0] * n
    return [(v - mu) / sigma for v in values]
