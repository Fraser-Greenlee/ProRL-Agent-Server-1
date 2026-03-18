from __future__ import annotations

import pytest

from trajectory.builder.prefix_merging import (
    PrefixMergingBuilder,
    _edit_distance_within_budget,
    _sequence_similarity,
)
from trajectory.models import CompletionSession
from trajectory.registry import BuilderRegistry


def _make_completion(
    *,
    completion_id: str,
    timestamp: str,
    prompt_ids: list[int],
    response_ids: list[int],
    prompt_content: str,
    response_content: str,
    include_logprobs: bool = False,
) -> dict:
    choice: dict = {
        "message": {"role": "assistant", "content": response_content},
        "finish_reason": "stop",
        "token_ids": response_ids,
    }
    if include_logprobs:
        choice["logprobs"] = {
            "content": [{"token_id": token_id, "logprob": -0.1} for token_id in response_ids]
        }

    return {
        "completion_id": completion_id,
        "timestamp": timestamp,
        "request": {"messages": [{"role": "user", "content": prompt_content}]},
        "response": {
            "prompt_token_ids": prompt_ids,
            "choices": [choice],
        },
    }


@pytest.mark.asyncio
async def test_prefix_merging_merges_contiguous_chain_and_splits_on_context_shift() -> None:
    session = CompletionSession.model_validate(
        {
            "session_id": "sess-merge",
            "completions": [
                _make_completion(
                    completion_id="b-1",
                    timestamp="2026-03-17T00:00:04+00:00",
                    prompt_ids=[9],
                    response_ids=[10],
                    prompt_content="branch b",
                    response_content="b2",
                ),
                _make_completion(
                    completion_id="a-2",
                    timestamp="2026-03-17T00:00:02+00:00",
                    prompt_ids=[1, 2],
                    response_ids=[3],
                    prompt_content="branch a follow-up",
                    response_content="a3",
                    include_logprobs=True,
                ),
                _make_completion(
                    completion_id="a-3",
                    timestamp="2026-03-17T00:00:03+00:00",
                    prompt_ids=[1, 2, 4],
                    response_ids=[5],
                    prompt_content="branch a compacted",
                    response_content="a5",
                ),
                _make_completion(
                    completion_id="a-1",
                    timestamp="2026-03-17T00:00:01+00:00",
                    prompt_ids=[1],
                    response_ids=[2],
                    prompt_content="branch a",
                    response_content="a2",
                    include_logprobs=True,
                ),
            ],
        }
    )

    trajectory = await PrefixMergingBuilder().build(session)

    assert [trace.prompt_ids for trace in trajectory.traces] == [[1], [1, 2, 4], [9]]
    assert [trace.response_ids for trace in trajectory.traces] == [[2, 3], [5], [10]]
    assert trajectory.traces[0].prompt_messages == [{"role": "user", "content": "branch a"}]
    assert trajectory.traces[0].response_messages == [
        {"role": "assistant", "content": "a2"},
        {"role": "assistant", "content": "a3"},
    ]
    assert trajectory.traces[0].response_logprobs == [
        {"token_id": 2, "logprob": -0.1},
        {"token_id": 3, "logprob": -0.1},
    ]
    assert trajectory.traces[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_prefix_merging_keeps_parallel_first_turns_separate() -> None:
    session = CompletionSession.model_validate(
        {
            "session_id": "sess-parallel",
            "completions": [
                _make_completion(
                    completion_id="p-1",
                    timestamp="2026-03-17T00:00:01+00:00",
                    prompt_ids=[7],
                    response_ids=[8],
                    prompt_content="shared root",
                    response_content="left",
                ),
                _make_completion(
                    completion_id="p-2",
                    timestamp="2026-03-17T00:00:02+00:00",
                    prompt_ids=[7],
                    response_ids=[9],
                    prompt_content="shared root",
                    response_content="right",
                ),
                _make_completion(
                    completion_id="p-3",
                    timestamp="2026-03-17T00:00:03+00:00",
                    prompt_ids=[7, 8],
                    response_ids=[10],
                    prompt_content="left follow-up",
                    response_content="left-2",
                ),
                _make_completion(
                    completion_id="p-4",
                    timestamp="2026-03-17T00:00:04+00:00",
                    prompt_ids=[7, 9],
                    response_ids=[11],
                    prompt_content="right follow-up",
                    response_content="right-2",
                ),
            ],
        }
    )

    trajectory = await PrefixMergingBuilder().build(session)

    assert len(trajectory.traces) == 2
    assert [trace.prompt_ids for trace in trajectory.traces] == [[7], [7]]
    assert [trace.response_ids for trace in trajectory.traces] == [[8, 10], [9, 11]]


@pytest.mark.asyncio
async def test_fuzzy_match_chains_with_minor_token_diff() -> None:
    """A single substituted token in a 103-length prefix is within 0.99 tolerance."""
    base_prompt = list(range(100))
    response = [100, 101, 102]
    # Next turn's prompt: same as base_prompt + response, but token 50 is tweaked.
    tweaked_prompt = list(range(50)) + [999] + list(range(51, 100)) + response

    session = CompletionSession.model_validate(
        {
            "session_id": "sess-fuzzy",
            "completions": [
                _make_completion(
                    completion_id="f-1",
                    timestamp="2026-03-17T00:00:01+00:00",
                    prompt_ids=base_prompt,
                    response_ids=response,
                    prompt_content="turn 1",
                    response_content="r1",
                ),
                _make_completion(
                    completion_id="f-2",
                    timestamp="2026-03-17T00:00:02+00:00",
                    prompt_ids=tweaked_prompt,
                    response_ids=[200, 201],
                    prompt_content="turn 2 with tweak",
                    response_content="r2",
                ),
            ],
        }
    )

    exact = await PrefixMergingBuilder().build(session)
    assert len(exact.traces) == 2, "exact match must not merge tweaked prompt"

    fuzzy = await PrefixMergingBuilder(match_tolerance=0.99).build(session)
    assert len(fuzzy.traces) == 1
    assert fuzzy.traces[0].response_ids == [100, 101, 102, 200, 201]
    assert fuzzy.metadata["match_tolerance"] == 0.99


@pytest.mark.asyncio
async def test_fuzzy_match_rejects_when_below_tolerance() -> None:
    """Two substitutions in a 12-token prefix (~0.83 similarity) fail at tolerance=0.99."""
    session = CompletionSession.model_validate(
        {
            "session_id": "sess-reject",
            "completions": [
                _make_completion(
                    completion_id="r-1",
                    timestamp="2026-03-17T00:00:01+00:00",
                    prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    response_ids=[11, 12],
                    prompt_content="turn 1",
                    response_content="r1",
                ),
                _make_completion(
                    completion_id="r-2",
                    timestamp="2026-03-17T00:00:02+00:00",
                    prompt_ids=[1, 2, 99, 4, 5, 6, 7, 8, 99, 10, 11, 12],
                    response_ids=[20],
                    prompt_content="turn 2",
                    response_content="r2",
                ),
            ],
        }
    )

    strict = await PrefixMergingBuilder(match_tolerance=0.99).build(session)
    assert len(strict.traces) == 2

    relaxed = await PrefixMergingBuilder(match_tolerance=0.8).build(session)
    assert len(relaxed.traces) == 1


def test_match_tolerance_validation() -> None:
    with pytest.raises(ValueError, match="match_tolerance"):
        PrefixMergingBuilder(match_tolerance=0.0)
    with pytest.raises(ValueError, match="match_tolerance"):
        PrefixMergingBuilder(match_tolerance=1.5)


# ---------------------------------------------------------------------------
# Unit tests for the similarity helpers
# ---------------------------------------------------------------------------

class TestEditDistanceWithinBudget:
    def test_identical(self) -> None:
        assert _edit_distance_within_budget((1, 2, 3), (1, 2, 3), 0) == 0

    def test_single_substitution(self) -> None:
        assert _edit_distance_within_budget((1, 2, 3), (1, 9, 3), 1) == 1

    def test_over_budget_returns_none(self) -> None:
        assert _edit_distance_within_budget((1, 2, 3), (1, 9, 3), 0) is None

    def test_insertion(self) -> None:
        assert _edit_distance_within_budget((1, 2, 3), (1, 2, 5, 3), 1) == 1

    def test_deletion(self) -> None:
        assert _edit_distance_within_budget((1, 2, 3), (1, 3), 1) == 1

    def test_empty_sequences(self) -> None:
        assert _edit_distance_within_budget((), (), 0) == 0
        assert _edit_distance_within_budget((), (1,), 1) == 1
        assert _edit_distance_within_budget((), (1, 2), 1) is None

    def test_length_diff_exceeds_budget(self) -> None:
        assert _edit_distance_within_budget((1,), (1, 2, 3, 4), 2) is None


class TestSequenceSimilarity:
    def test_identical(self) -> None:
        assert _sequence_similarity((1, 2, 3), (1, 2, 3)) == 1.0

    def test_completely_different(self) -> None:
        assert _sequence_similarity((1, 2, 3), (4, 5, 6)) == pytest.approx(0.0)

    def test_one_edit_in_hundred(self) -> None:
        a = tuple(range(100))
        b = tuple(range(50)) + (999,) + tuple(range(51, 100))
        assert _sequence_similarity(a, b) == pytest.approx(0.99)

    def test_both_empty(self) -> None:
        assert _sequence_similarity((), ()) == 1.0

    def test_min_ratio_prunes(self) -> None:
        a = (1, 2, 3)
        b = (4, 5, 6)
        assert _sequence_similarity(a, b, min_ratio=0.99) == 0.0


def test_builder_registry_registers_prefix_merging() -> None:
    registry = BuilderRegistry()

    assert "prefix_merging" in registry.list_builders()
    assert isinstance(registry.get_builder("prefix_merging"), PrefixMergingBuilder)
