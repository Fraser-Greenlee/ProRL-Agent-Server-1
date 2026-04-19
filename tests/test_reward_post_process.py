"""Unit tests for the trajectory-aware reward post-processor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from slime_bridge.reward_post_process import post_process_rewards


@dataclass
class _FakeSample:
    group_index: int
    index: int
    _reward: float

    def get_reward_value(self, args):  # noqa: ARG002  (match Slime signature)
        return self._reward


def _samples(entries: list[tuple[int, int, float]]) -> list[_FakeSample]:
    return [_FakeSample(g, i, r) for g, i, r in entries]


def _args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Shared-within-trajectory, normalized-across-trajectories-within-group
# ---------------------------------------------------------------------------


def test_shared_within_trajectory_and_normalized_across_trajectories() -> None:
    # Group 0:  traj 0 → 3 traces reward 1.0 ; traj 1 → 1 trace reward 0.0
    # Group 1:  traj 0 → 2 traces reward 0.5 ; traj 1 → 2 traces reward 0.5
    samples = _samples(
        [
            (0, 0, 1.0), (0, 0, 1.0), (0, 0, 1.0),
            (0, 1, 0.0),
            (1, 0, 0.5), (1, 0, 0.5),
            (1, 1, 0.5), (1, 1, 0.5),
        ]
    )
    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [1.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]

    # Group 0: two trajectories (1.0, 0.0), torch sample std (N-1):
    # std = 0.7071 → rewards = +0.7071, -0.7071
    g0_traj0 = rewards[:3]
    g0_traj1 = rewards[3:4]
    assert len(set(g0_traj0)) == 1  # shared within trajectory
    assert math.isclose(g0_traj0[0], math.sqrt(2) / 2, rel_tol=1e-5)
    assert math.isclose(g0_traj1[0], -math.sqrt(2) / 2, rel_tol=1e-5)

    # Group 1: two trajectories both 0.5 → std=0, falls to zero tensor
    for v in rewards[4:]:
        assert v == 0.0


def test_single_trace_per_trajectory_reduces_to_group_normalization() -> None:
    # Each sample has a unique index → behaves like plain GRPO group-norm.
    samples = _samples([(0, 0, 1.0), (0, 1, 0.0), (0, 2, 0.0)])
    raw, rewards = post_process_rewards(_args(), samples)
    assert raw == [1.0, 0.0, 0.0]
    # Mean 1/3, torch sample std (N-1) = sqrt(1/3) → centered/std
    mean = sum([1.0, 0.0, 0.0]) / 3
    std = math.sqrt(((1 - mean) ** 2 + 2 * (0 - mean) ** 2) / 2)  # N-1
    expected = [(1.0 - mean) / (std + 1e-6), -mean / (std + 1e-6), -mean / (std + 1e-6)]
    for got, want in zip(rewards, expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-4)


def test_std_normalization_disabled_mean_centers_only() -> None:
    samples = _samples([(0, 0, 1.0), (0, 0, 1.0), (0, 1, 0.0)])
    raw, rewards = post_process_rewards(
        _args(grpo_std_normalization=False), samples
    )
    assert raw == [1.0, 1.0, 0.0]
    # Two trajectories: 1.0, 0.0 → centered: +0.5, -0.5
    assert math.isclose(rewards[0], 0.5, rel_tol=1e-6)
    assert math.isclose(rewards[1], 0.5, rel_tol=1e-6)
    assert math.isclose(rewards[2], -0.5, rel_tol=1e-6)


def test_normalization_disabled_returns_raw() -> None:
    samples = _samples([(0, 0, 1.0), (0, 1, 0.0)])
    raw, rewards = post_process_rewards(
        _args(rewards_normalization=False), samples
    )
    assert raw == rewards == [1.0, 0.0]


def test_unsupported_estimator_returns_raw() -> None:
    samples = _samples([(0, 0, 1.0), (0, 1, 0.0)])
    raw, rewards = post_process_rewards(_args(advantage_estimator="ppo"), samples)
    assert raw == rewards == [1.0, 0.0]


def test_constant_group_zeroes_with_std_normalization() -> None:
    # Two trajectories with identical rewards → std=0 path → zeros.
    samples = _samples([(0, 0, 0.7), (0, 1, 0.7)])
    raw, rewards = post_process_rewards(_args(), samples)
    assert raw == [0.7, 0.7]
    assert rewards == [0.0, 0.0]


def test_first_reward_per_trajectory_wins_on_conflicting_inputs() -> None:
    # Safety net: if two samples tagged to the same trajectory carry
    # different rewards, the first wins.  Adapter guarantees consistency,
    # but the function shouldn't crash if an evaluator ever drifted.
    samples = _samples([(0, 0, 1.0), (0, 0, 0.0), (0, 1, 0.5)])
    raw, rewards = post_process_rewards(
        _args(grpo_std_normalization=False), samples
    )
    assert raw == [1.0, 0.0, 0.5]
    # Trajectories: 1.0 (first-seen) and 0.5 → mean 0.75 → +0.25, +0.25, -0.25
    assert math.isclose(rewards[0], 0.25, rel_tol=1e-6)
    assert math.isclose(rewards[1], 0.25, rel_tol=1e-6)
    assert math.isclose(rewards[2], -0.25, rel_tol=1e-6)
