"""Base interface for advantage estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from polar.trajectory.models import Trajectory


class BaseAdvantageEstimator(ABC):
    """Strategy plugin that computes per-trace advantages for a group of trajectories.

    A "group" is all trajectories sampled for the same task/problem.
    Each trajectory may contain a variable number of traces, each with
    its own ``reward`` set by an evaluator.  The estimator writes
    ``trace.advantage`` on every trace and returns the updated list.

    Estimators are instantiated per-request with ``**config`` from the
    :class:`~polar.trajectory.models.StrategySpec`.
    """

    def __init__(self, **config: Any) -> None:  # noqa: B027
        ...

    @abstractmethod
    def estimate(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Compute advantages in-place and return the (mutated) trajectory list.

        Traces with ``reward is None`` should be skipped or assigned zero
        advantage.  Implementations may assume that all trajectories
        belong to the same task group.
        """
