"""Smoke tests for the built-in advantage estimators."""

from __future__ import annotations

import math

import pytest

from polar.trajectory.adv_estimator import (
    HierarchicalGroupAdvantageEstimator,
    ShareAdvInGroupAdvantageEstimator,
)
from polar.trajectory.models import Trace, Trajectory
from polar.trajectory.registry import default_adv_estimator_registry


def _traj(rewards: list[float | None]) -> Trajectory:
    return Trajectory(
        status="COMPLETED",
        traces=[Trace(reward=r) for r in rewards],
    )


# ---------------------------------------------------------------------------
# Registry exposes both built-in estimators
# ---------------------------------------------------------------------------


def test_registry_lists_both_estimators() -> None:
    registry = default_adv_estimator_registry()
    assert set(registry.list_strategies()) == {
        "hierarchical_group",
        "share_adv_in_group",
    }


# ---------------------------------------------------------------------------
# share_adv_in_group: identical-reward requirement + shared advantage
# ---------------------------------------------------------------------------


def test_share_adv_in_group_rejects_unequal_rewards_in_one_trajectory() -> None:
    estimator = ShareAdvInGroupAdvantageEstimator()
    trajectories = [_traj([1.0, 0.0])]
    with pytest.raises(ValueError, match="single shared reward"):
        estimator.estimate(trajectories)


def test_share_adv_in_group_broadcasts_group_advantage() -> None:
    # Three trajectories: rewards 1.0, 0.0, 0.0 — standardize → ~+1.41, ~-0.71, ~-0.71.
    estimator = ShareAdvInGroupAdvantageEstimator()
    trajectories = [_traj([1.0, 1.0]), _traj([0.0]), _traj([0.0, 0.0, 0.0])]
    result = estimator.estimate(trajectories)

    advs_per_traj = [[trace.advantage for trace in traj.traces] for traj in result]
    # All traces within one trajectory must share exactly the same advantage.
    for advs in advs_per_traj:
        assert len(set(advs)) == 1
    # Between-trajectory follows population-std standardization.
    first = advs_per_traj[0][0]
    assert math.isclose(first, math.sqrt(2), rel_tol=1e-6)
    for advs in advs_per_traj[1:]:
        assert math.isclose(advs[0], -math.sqrt(2) / 2, rel_tol=1e-6)


def test_share_adv_in_group_zero_variance_returns_zeros() -> None:
    estimator = ShareAdvInGroupAdvantageEstimator()
    trajectories = [_traj([0.5]), _traj([0.5, 0.5])]
    result = estimator.estimate(trajectories)
    for traj in result:
        for trace in traj.traces:
            assert trace.advantage == 0.0


def test_share_adv_in_group_empty_input_is_noop() -> None:
    estimator = ShareAdvInGroupAdvantageEstimator()
    assert estimator.estimate([]) == []


def test_share_adv_in_group_accepts_gateway_last_trace_reward_pattern() -> None:
    # The gateway writes `outcome_reward` onto the last trace only, leaving
    # prior traces with reward=None. The estimator must treat that as the
    # trajectory's shared reward, not as a conflict.
    estimator = ShareAdvInGroupAdvantageEstimator()
    trajectories = [_traj([None, None, 1.0]), _traj([None, 0.0])]
    result = estimator.estimate(trajectories)

    for traj in result:
        advs = {trace.advantage for trace in traj.traces}
        assert len(advs) == 1  # every trace inside a trajectory shares one advantage
    assert result[0].traces[0].advantage > 0
    assert result[1].traces[0].advantage < 0


# ---------------------------------------------------------------------------
# hierarchical_group still works after the shared-standardize refactor
# ---------------------------------------------------------------------------


def test_hierarchical_group_assigns_advantages() -> None:
    estimator = HierarchicalGroupAdvantageEstimator(beta=0.0)
    trajectories = [_traj([1.0, 1.0]), _traj([0.0, 0.0])]
    result = estimator.estimate(trajectories)
    # beta=0 → within-trajectory term drops out; each trace gets A_between / K_i.
    for traj in result:
        advs = {trace.advantage for trace in traj.traces}
        assert len(advs) == 1
