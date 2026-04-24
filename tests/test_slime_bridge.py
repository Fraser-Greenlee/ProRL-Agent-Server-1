"""Smoke tests for the slime_bridge extraction.

The slime_bridge package is the consumer-side adapter between Slime and
Polar — it depends on Polar but nothing in Polar depends back. These
tests cover:

- the package lives at ``slime_bridge`` (not ``polar.slime``);
- the shared message-flattening helpers are single-sourced;
- ``reward_func`` reads the reward already embedded in Polar samples;
- ``render_task_payload`` / ``render_instruction`` resolve sample /
  metadata placeholders;
- ``session_result_to_samples`` drops traces without tokens.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from slime_bridge._messages import (
    flatten_content,
    messages_to_text,
    prompt_to_instruction_text,
)


# ---------------------------------------------------------------------------
# Package move
# ---------------------------------------------------------------------------


def test_polar_slime_module_is_gone() -> None:
    with pytest.raises(ImportError):
        import polar.slime  # noqa: F401


def test_slime_bridge_package_importable() -> None:
    import slime_bridge  # noqa: F401
    import slime_bridge.adapter  # noqa: F401
    import slime_bridge.config  # noqa: F401
    import slime_bridge.reward  # noqa: F401
    import slime_bridge.rollout  # noqa: F401


# ---------------------------------------------------------------------------
# Shared message helpers (single-sourced in _messages.py)
# ---------------------------------------------------------------------------


def test_flatten_content_handles_str_list_and_none() -> None:
    assert flatten_content("hello") == "hello"
    assert flatten_content([{"type": "text", "text": "hi "}, {"text": "there"}]) == "hi there"
    assert flatten_content(None) == ""
    assert flatten_content(42) == "42"


def test_prompt_to_instruction_text_renders_chat_list() -> None:
    prompt = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    assert prompt_to_instruction_text(prompt) == "[system] sys\n\n[user] u"


def test_prompt_to_instruction_text_passes_str_through() -> None:
    assert prompt_to_instruction_text("raw prompt") == "raw prompt"


def test_messages_to_text_drops_empty_content() -> None:
    messages = [
        {"role": "assistant", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
    ]
    assert messages_to_text(messages) == "[assistant] hi"


def test_slime_bridge_does_not_redefine_flatten_helpers() -> None:
    """Guard against re-introducing duplicated helpers in rollout.py/adapter.py.

    The pre-cleanup state had _prompt_to_instruction_text/_flatten_content
    duplicated in both rollout.py and adapter.py. Any reintroduction should
    fail this test.
    """
    from slime_bridge import adapter, rollout

    for module in (adapter, rollout):
        for name in ("_prompt_to_instruction_text", "_flatten_content"):
            assert not hasattr(module, name), (
                f"{module.__name__}.{name} was reintroduced; "
                f"use slime_bridge._messages instead"
            )


# ---------------------------------------------------------------------------
# reward_func
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_reward_func_reads_preset_reward_dict() -> None:
    from slime_bridge.reward import reward_func

    sample = SimpleNamespace(reward={"score": 0.75})
    args = SimpleNamespace()
    result = _run(reward_func(args, sample))
    assert result == {"score": 0.75}


def test_reward_func_handles_list_of_samples() -> None:
    from slime_bridge.reward import reward_func

    samples = [
        SimpleNamespace(reward={"score": 1.0}),
        SimpleNamespace(reward={"score": 0.0}),
    ]
    args = SimpleNamespace()
    result = _run(reward_func(args, samples))
    assert result == [{"score": 1.0}, {"score": 0.0}]


def test_reward_func_respects_custom_reward_key() -> None:
    from slime_bridge.reward import reward_func

    sample = SimpleNamespace(reward={"score": 0.5})
    args = SimpleNamespace(polar_reward_key="my_reward")
    assert _run(reward_func(args, sample)) == {"my_reward": 0.5}


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def test_render_task_payload_resolves_sample_metadata_placeholders() -> None:
    from slime_bridge.config import PolarSlimeConfig, render_task_payload

    config = PolarSlimeConfig(
        rollout_server_url="http://rollout",
        task_template={
            "agent": {"harness": "claude_code"},
            "metadata": {"instance_id": "{sample.metadata.instance_id}"},
        },
        task_id_template="polar-{rollout_id}-{sample.group_index}",
        instruction_template=None,
        reward_key="score",
        max_concurrency=1,
        max_session_concurrency=1,
        max_async_level=2,
        max_off_policy_steps=1,
        max_task_retries=2,
        request_timeout=None,
        callback_host="127.0.0.1",
        scoring_mode="group",
        tokenizer_name_or_path=None,
        add_generation_prompt=True,
        eval_dataset_name="polar_eval",
    )
    sample = SimpleNamespace(
        group_index=3,
        metadata={"instance_id": "django__django-11001"},
    )
    payload = render_task_payload(
        args=SimpleNamespace(sglang_router_ip=None, sglang_router_port=None),
        config=config,
        sample=sample,
        instruction="solve it",
        rollout_id=7,
        task_position=0,
        num_rollouts=4,
    )
    assert payload["task_id"] == "polar-7-3"
    assert payload["instruction"] == "solve it"
    assert payload["num_samples"] == 4
    assert payload["metadata"]["instance_id"] == "django__django-11001"


def test_render_instruction_falls_back_to_raw_prompt_when_no_template() -> None:
    from slime_bridge.config import PolarSlimeConfig, render_instruction

    config = PolarSlimeConfig(
        rollout_server_url="http://rollout",
        task_template={"agent": {"harness": "shell"}},
        task_id_template="polar-{rollout_id}",
        instruction_template=None,
        reward_key="score",
        max_concurrency=1,
        max_session_concurrency=1,
        max_async_level=2,
        max_off_policy_steps=1,
        max_task_retries=2,
        request_timeout=None,
        callback_host="127.0.0.1",
        scoring_mode="group",
        tokenizer_name_or_path=None,
        add_generation_prompt=True,
        eval_dataset_name="polar_eval",
    )
    result = render_instruction(
        args=SimpleNamespace(sglang_router_ip=None, sglang_router_port=None),
        config=config,
        sample=SimpleNamespace(metadata={}),
        prompt_text="raw prompt",
        rollout_id=0,
        task_position=0,
        num_rollouts=1,
    )
    assert result == "raw prompt"


# ---------------------------------------------------------------------------
# Adapter — session_result_to_samples drops traces without tokens
# ---------------------------------------------------------------------------


class _FakeSampleStatus:
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    TRUNCATED = "TRUNCATED"
    COMPLETED = "COMPLETED"


class _FakeSample:
    Status = _FakeSampleStatus

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_session_result_to_samples_drops_empty_token_traces(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)

    trajectory = Trajectory(
        status="COMPLETED",
        traces=[
            Trace(
                prompt_ids=[1, 2],
                response_ids=[3, 4],
                response_logprobs=[
                    {"token": "a", "token_id": 3, "logprob": -0.3},
                    {"token": "b", "token_id": 4, "logprob": -0.4},
                ],
                finish_reason="stop",
            ),
            Trace(prompt_ids=[], response_ids=[], finish_reason="stop"),
        ],
    )
    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=trajectory,
        timing=SessionTiming(),
    )
    samples = adapter.session_result_to_samples(
        result, group_index=0, trajectory_index=7
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [1, 2, 3, 4]
    assert sample.response_length == 2
    assert sample.status == _FakeSampleStatus.COMPLETED
    assert sample.reward == {"score": 0.0}
    assert sample.index == 7


# ---------------------------------------------------------------------------
# Adapter — loss_mask masks canonical interstitials but keeps real assistant
# tokens even when their logprob is exactly 0 (high-confidence predictions).
# ---------------------------------------------------------------------------


def test_session_result_to_samples_masks_canonical_interstitial(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)

    # Mixed response: real assistant (real logprob), real assistant (logprob=0
    # but still the model's own sample), canonical interstitial (no "token"
    # field), real assistant again.
    response_logprobs = [
        {"token": "<a>", "token_id": 10, "logprob": -0.5},
        {"token": "<b>", "token_id": 11, "logprob": 0.0},       # legit p=1 sample
        {"token_id": 12, "logprob": 0.0},                        # interstitial
        {"token_id": 13, "logprob": 0.0},                        # interstitial
        {"token": "<c>", "token_id": 14, "logprob": -0.2},
    ]
    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[10, 11, 12, 13, 14],
        response_logprobs=response_logprobs,
        finish_reason="stop",
    )
    trajectory = Trajectory(status="COMPLETED", traces=[trace])
    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=trajectory,
        timing=SessionTiming(),
    )

    samples = adapter.session_result_to_samples(
        result, group_index=0, trajectory_index=0
    )
    assert len(samples) == 1
    sample = samples[0]
    # Real assistant positions (including the logprob=0 one) stay trainable;
    # interstitial positions are masked out.
    assert sample.loss_mask == [1, 1, 0, 0, 1]
    # Rollout logprobs still span every position (trainer needs them aligned).
    assert sample.rollout_log_probs == [-0.5, 0.0, 0.0, 0.0, -0.2]


def test_session_result_to_samples_requires_logprobs_for_trainable_trace(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)

    trace = Trace(prompt_ids=[1], response_ids=[10, 11, 12], finish_reason="stop")
    trajectory = Trajectory(status="COMPLETED", traces=[trace])
    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=trajectory,
        timing=SessionTiming(),
    )
    with pytest.raises(adapter.RolloutLogprobError):
        adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)


def test_dummy_placeholder_is_fully_masked(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trajectory
    from slime_bridge import adapter

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)

    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.ERROR,
        trajectory=Trajectory(status="ERROR", traces=[], error="no trace"),
        timing=SessionTiming(),
    )
    samples = adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)
    assert len(samples) == 1
    assert samples[0].loss_mask == [0]
    assert samples[0].rollout_log_probs == [0.0]
    assert samples[0].remove_sample is True


# ---------------------------------------------------------------------------
# Async rollout worker buffering / staleness
# ---------------------------------------------------------------------------


def _worker_args(**overrides: Any) -> SimpleNamespace:
    values = {
        "polar_rollout_url": "http://rollout",
        "polar_task_template": {"agent": {"harness": "shell"}},
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "update_weights_interval": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _NoopDataSource:
    def get_samples(self, num_samples: int) -> list[list[Any]]:
        del num_samples
        return []


def test_async_worker_keeps_completed_overflow() -> None:
    from slime_bridge.rollout import AsyncPolarRolloutWorker, _CompletedGroup

    worker = AsyncPolarRolloutWorker(_worker_args(), _NoopDataSource())
    worker.output_queue.put(
        _CompletedGroup(
            group_id=1,
            group=[object()],
            samples=[_FakeSample(metadata={})],
            task_id="t1",
            attempt=0,
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )
    worker.output_queue.put(
        _CompletedGroup(
            group_id=2,
            group=[object()],
            samples=[_FakeSample(metadata={})],
            task_id="t2",
            attempt=0,
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )

    first = worker.drain_completed(max_groups=1, rollout_id=0)
    second = worker.drain_completed(max_groups=1, rollout_id=0)

    assert [item.group_id for item in first] == [1]
    assert [item.group_id for item in second] == [2]


def test_async_worker_requeues_too_stale_completed_group() -> None:
    from slime_bridge.rollout import AsyncPolarRolloutWorker, _CompletedGroup

    worker = AsyncPolarRolloutWorker(
        _worker_args(polar_max_off_policy_steps=0, polar_max_task_retries=1),
        _NoopDataSource(),
    )
    group = [object()]
    worker.output_queue.put(
        _CompletedGroup(
            group_id=1,
            group=group,
            samples=["stale"],
            task_id="t1",
            attempt=0,
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )

    assert worker.drain_completed(max_groups=1, rollout_id=1) == []
    retry = worker.retry_queue.get_nowait()
    assert retry.group is group
    assert retry.attempt == 1
