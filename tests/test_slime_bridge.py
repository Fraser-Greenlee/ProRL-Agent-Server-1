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


def test_session_result_to_samples_tokenizes_prompt_messages_when_prompt_ids_missing(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is True
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            assert messages == [{"role": "user", "content": "hi"}]
            return {"input_ids": [101, 102]}

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda _: _FakeTokenizer())

    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(
            status="COMPLETED",
            traces=[
                Trace(
                    prompt_ids=[],
                    prompt_messages=[{"role": "user", "content": "hi"}],
                    response_ids=[201],
                    response_logprobs=[
                        {"token": "ok", "token_id": 201, "logprob": -0.1},
                    ],
                    finish_reason="stop",
                )
            ],
        ),
        timing=SessionTiming(),
    )

    samples = adapter.session_result_to_samples(
        result,
        group_index=0,
        trajectory_index=0,
        tokenizer_name_or_path="/tmp/tokenizer",
    )

    assert len(samples) == 1
    assert samples[0].tokens == [101, 102, 201]
    assert samples[0].response_length == 1
    assert samples[0].loss_mask == [1]


def test_session_result_to_samples_flattens_openai_text_parts_for_chat_template(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "user", "content": "hello\nworld"},
                {"role": "assistant", "content": ""},
            ]
            return [11, 12]

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda _: _FakeTokenizer())

    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(
            status="COMPLETED",
            traces=[
                Trace(
                    prompt_ids=[],
                    prompt_messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "text", "text": "\nworld"},
                            ],
                        },
                        {"role": "assistant", "content": None},
                    ],
                    response_ids=[13],
                    response_logprobs=[
                        {"token": "!", "token_id": 13, "logprob": -0.1},
                    ],
                    finish_reason="stop",
                )
            ],
        ),
        timing=SessionTiming(),
    )

    samples = adapter.session_result_to_samples(
        result,
        group_index=0,
        trajectory_index=0,
        tokenizer_name_or_path="/tmp/tokenizer",
    )

    assert len(samples) == 1
    assert samples[0].tokens == [11, 12, 13]


def test_session_result_to_samples_normalizes_tool_messages_for_chat_template(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [
                {"role": "assistant", "content": "checking\n[tool call: rg] files"},
                {"role": "user", "content": "[tool result: call_1]\nfound"},
            ]
            return [21, 22]

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda _: _FakeTokenizer())

    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(
            status="COMPLETED",
            traces=[
                Trace(
                    prompt_ids=[],
                    prompt_messages=[
                        {
                            "role": "assistant",
                            "content": "checking",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"name": "rg", "arguments": "files"},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": [{"type": "text", "text": "found"}],
                        },
                    ],
                    response_ids=[23],
                    response_logprobs=[
                        {"token": "done", "token_id": 23, "logprob": -0.1},
                    ],
                    finish_reason="stop",
                )
            ],
        ),
        timing=SessionTiming(),
    )

    samples = adapter.session_result_to_samples(
        result,
        group_index=0,
        trajectory_index=0,
        tokenizer_name_or_path="/tmp/tokenizer",
    )

    assert len(samples) == 1
    assert samples[0].tokens == [21, 22, 23]


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


def test_session_result_to_samples_accepts_internal_trainable_marker(monkeypatch) -> None:
    from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
    from polar.trajectory.models import Trace, Trajectory
    from slime_bridge import adapter

    monkeypatch.setattr(adapter, "_load_sample_type", lambda: _FakeSample)

    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[10, 11, 12],
        response_logprobs=[
            {"token_id": 10, "logprob": -0.5, "_polar_trainable": True},
            {"token_id": 11, "logprob": 0.0},
            {"token_id": 12, "logprob": -0.2, "_polar_trainable": True},
        ],
        finish_reason="stop",
    )
    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(status="COMPLETED", traces=[trace]),
        timing=SessionTiming(),
    )

    samples = adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)
    assert samples[0].loss_mask == [1, 0, 1]
    assert samples[0].rollout_log_probs == [-0.5, 0.0, -0.2]


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


def test_config_defaults_off_policy_bound_to_async_level_plus_update_interval() -> None:
    from slime_bridge.config import resolve_polar_slime_config

    config = resolve_polar_slime_config(
        _worker_args(polar_max_async_level=3, update_weights_interval=1)
    )
    assert config.max_off_policy_steps == 4


def test_ceil_epoch_data_source_rounds_up_to_full_batch() -> None:
    from slime_bridge.data_source import ceil_to_batch_size

    assert ceil_to_batch_size(293, 8) == 296
    assert ceil_to_batch_size(288, 8) == 288


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
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )

    first = worker.drain_completed(max_groups=1, rollout_id=0)
    second = worker.drain_completed(max_groups=1, rollout_id=0)

    assert [item.group_id for item in first] == [1]
    assert [item.group_id for item in second] == [2]


def test_async_worker_drops_stale_completed_group(caplog: pytest.LogCaptureFixture) -> None:
    from slime_bridge.rollout import AsyncPolarRolloutWorker, _CompletedGroup

    worker = AsyncPolarRolloutWorker(
        _worker_args(polar_max_off_policy_steps=0),
        _NoopDataSource(),
    )
    worker.output_queue.put(
        _CompletedGroup(
            group_id=2,
            group=[object()],
            samples=["stale"],
            task_id="t2",
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )

    caplog.set_level("WARNING", logger="slime_bridge.rollout")
    assert worker.drain_completed(max_groups=1, rollout_id=1) == []
    assert worker.deferred_queue.empty()
    metrics = worker.snapshot_metrics()
    assert metrics["polar/dropped_groups"] == 1.0
    assert metrics["polar/dropped_stale_groups"] == 1.0
    assert metrics["polar/dropped_sessions"] == 1.0
    assert "Dropping stale Polar group 2 task=t2" in caplog.text


def test_async_worker_rejects_group_with_no_trainable_tokens(monkeypatch) -> None:
    from slime_bridge import rollout

    worker = rollout.AsyncPolarRolloutWorker(_worker_args(), _NoopDataSource())
    pending = rollout._PendingGroup(
        group_id=1,
        group=[object()],
        submitted_rollout_id=0,
        policy_version=0,
        session_cost=1,
    )
    task_result = SimpleNamespace(
        task_id="t1",
        status="completed",
        results=[
            SimpleNamespace(
                session_id="s1",
                status="COMPLETED",
                trajectory=SimpleNamespace(status="COMPLETED"),
            )
        ],
        result_paths=[],
    )

    async def fake_submit_with_callback(client: Any, payload: dict[str, Any]) -> Any:
        del client, payload
        return task_result

    monkeypatch.setattr(
        rollout,
        "_build_task_payload",
        lambda **kwargs: {"task_id": "t1"},
    )
    monkeypatch.setattr(
        rollout,
        "_convert_task_result_to_samples",
        lambda *args, **kwargs: [
            _FakeSample(
                loss_mask=[0, 0],
                response_length=2,
                remove_sample=False,
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(worker, "_submit_with_callback", fake_submit_with_callback)

    with pytest.raises(rollout.PolarRolloutSchedulerError, match="zero trainable tokens"):
        _run(worker._submit_attempt(object(), pending))


def test_async_worker_drops_zero_trainable_group(monkeypatch, caplog) -> None:
    from slime_bridge import rollout

    worker = rollout.AsyncPolarRolloutWorker(_worker_args(), _NoopDataSource())
    pending = rollout._PendingGroup(
        group_id=3,
        group=[object(), object()],
        submitted_rollout_id=0,
        policy_version=0,
        session_cost=2,
    )
    calls = 0

    async def fake_submit_attempt(client: Any, attempted: Any) -> Any:
        nonlocal calls
        del client, attempted
        calls += 1
        raise rollout.PolarRolloutSchedulerError("Task t3 produced zero trainable tokens")

    monkeypatch.setattr(worker, "_submit_attempt", fake_submit_attempt)
    caplog.set_level("WARNING", logger="slime_bridge.rollout")

    _run(worker._submit_and_collect(object(), pending))

    assert calls == 1
    assert worker.output_queue.empty()
    metrics = worker.snapshot_metrics()
    assert metrics["polar/dropped_groups"] == 1.0
    assert metrics["polar/dropped_zero_trainable_groups"] == 1.0
    assert metrics["polar/dropped_sessions"] == 2.0
    assert "Dropping Polar group 3 because of zero trainable tokens" in caplog.text


def test_async_worker_drops_failed_group(monkeypatch, caplog) -> None:
    from slime_bridge import rollout

    worker = rollout.AsyncPolarRolloutWorker(_worker_args(), _NoopDataSource())
    pending = rollout._PendingGroup(
        group_id=4,
        group=[object()],
        submitted_rollout_id=0,
        policy_version=0,
        session_cost=1,
    )

    async def fake_submit_attempt(client: Any, attempted: Any) -> Any:
        del client, attempted
        raise RuntimeError("gateway exploded")

    monkeypatch.setattr(worker, "_submit_attempt", fake_submit_attempt)
    caplog.set_level("WARNING", logger="slime_bridge.rollout")

    _run(worker._submit_and_collect(object(), pending))

    assert worker.output_queue.empty()
    metrics = worker.snapshot_metrics()
    assert metrics["polar/dropped_groups"] == 1.0
    assert metrics["polar/dropped_failed_groups"] == 1.0
    assert metrics["polar/dropped_sessions"] == 1.0
    assert "Dropping Polar group 4 because of task failure" in caplog.text
    assert "gateway exploded" in caplog.text


def test_eval_sample_groups_load_all_eval_rows(tmp_path, monkeypatch) -> None:
    from slime_bridge import rollout

    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"prompt": [{"role": "user", "content": "p0"}], "label": "", '
                '"metadata": {"instance_id": "i0"}}',
                '{"prompt": [{"role": "user", "content": "p1"}], "label": "", '
                '"metadata": {"instance_id": "i1"}}',
            ]
        )
        + "\n"
    )

    def inject_metadata(metadata: Any) -> dict[str, Any]:
        return {**metadata, "dataset": "eval"}

    dataset_cfg = SimpleNamespace(
        name="swegym_eval",
        path=str(path),
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        n_samples_per_eval_prompt=2,
        custom_generate_function_path=None,
        inject_metadata=inject_metadata,
    )
    monkeypatch.setattr(rollout, "_load_sample_type", lambda: _FakeSample)

    groups = rollout._load_eval_sample_groups(_worker_args(), dataset_cfg)

    assert len(groups) == 2
    assert [len(group) for group in groups] == [2, 2]
    assert [sample.index for group in groups for sample in group] == [0, 1, 2, 3]
    assert [group[0].group_index for group in groups] == [0, 1]
    assert groups[0][0].metadata == {"instance_id": "i0", "dataset": "eval"}
    assert groups[1][1].prompt == [{"role": "user", "content": "p1"}]


def test_eval_rollout_uses_eval_datasets_not_train_source(monkeypatch) -> None:
    from slime_bridge import rollout

    dataset_cfg = SimpleNamespace(name="swegym_eval", path="unused")
    sample_groups = [[_FakeSample(metadata={})]]

    class FailingTrainSource:
        def get_samples(self, num_samples: int) -> list[list[Any]]:
            raise AssertionError("eval should not consume training data_source")

    async def fake_submit_eval_groups(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        assert kwargs["sample_groups"] is sample_groups
        return {"rewards": [1.0], "truncated": [False], "samples": sample_groups[0]}, {
            "polar/task_count": 1,
        }

    monkeypatch.setattr(rollout, "_load_eval_sample_groups", lambda args, cfg: sample_groups)
    monkeypatch.setattr(rollout, "_submit_eval_groups", fake_submit_eval_groups)

    output = _run(
        rollout._run_eval_rollout(
            _worker_args(eval_datasets=[dataset_cfg]),
            rollout_id=4,
            data_source=FailingTrainSource(),
        )
    )

    assert output.data["swegym_eval"]["rewards"] == [1.0]
    assert output.metrics == {"polar/eval/swegym_eval/task_count": 1}


def test_eval_task_id_is_namespaced_away_from_train_ids() -> None:
    from slime_bridge import rollout

    train_task_id = "polar-swegym-grpo-11-11"
    eval_task_id = rollout._eval_task_id(
        train_task_id,
        dataset_name="swegym_eval",
        rollout_id=11,
        position=11,
    )

    assert eval_task_id != train_task_id
    assert eval_task_id == "polar-swegym-grpo-11-11-eval-swegym_eval-11-11"


def test_eval_metrics_use_completed_sessions_for_display_reward() -> None:
    from slime_bridge import rollout

    config = rollout.resolve_polar_slime_config(_worker_args())
    ok_sample = SimpleNamespace(
        reward={"score": 1.0},
        metadata={"polar": {"session_id": "s-ok", "session_status": "COMPLETED"}},
    )
    timeout_sample = SimpleNamespace(
        reward={"score": 0.0},
        metadata={
            "polar": {
                "session_id": "s-timeout",
                "session_status": "TIMEOUT",
                "placeholder": True,
            }
        },
    )
    task_result = SimpleNamespace(
        results=[
            SimpleNamespace(status="COMPLETED"),
            SimpleNamespace(status="TIMEOUT"),
        ]
    )

    metrics = rollout._build_metrics(
        config,
        [task_result],
        [[ok_sample, timeout_sample]],
        reward_filter="completed",
    )

    assert metrics["polar/completed_sessions"] == 1
    assert metrics["polar/timed_out_sessions"] == 1
    assert metrics["polar/reward_mean"] == 1.0
    assert metrics["polar/reward_completed_mean"] == 1.0
    assert metrics["polar/reward_all_mean"] == 0.5
    assert metrics["polar/reward_count"] == 1


def test_submit_and_wait_continues_after_transient_poll_read_error() -> None:
    import httpx
    from slime_bridge import rollout

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class FakeClient:
        def __init__(self) -> None:
            self.polls = 0

        async def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
            return FakeResponse({"task_id": "task-1"})

        async def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
            self.polls += 1
            if self.polls == 1:
                raise httpx.ReadError("transient")
            return FakeResponse(
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "total_sessions": 0,
                    "completed_sessions": 0,
                    "results": [],
                    "result_paths": [],
                }
            )

    result = _run(
        rollout._submit_and_wait_for_task(
            FakeClient(),
            "http://rollout",
            {"task_id": "task-1"},
            poll_interval=0,
        )
    )

    assert result.task_id == "task-1"
    assert result.status == "completed"


def test_async_worker_accepts_masked_bad_sessions_when_group_has_signal(monkeypatch) -> None:
    from slime_bridge import rollout

    worker = rollout.AsyncPolarRolloutWorker(_worker_args(), _NoopDataSource())
    pending = rollout._PendingGroup(
        group_id=1,
        group=[object(), object()],
        submitted_rollout_id=0,
        policy_version=0,
        session_cost=2,
    )
    task_result = SimpleNamespace(
        task_id="t1",
        status="completed",
        results=[
            SimpleNamespace(
                session_id="s-timeout",
                status="TIMEOUT",
                trajectory=SimpleNamespace(status="TIMEOUT"),
            ),
            SimpleNamespace(
                session_id="s-ok",
                status="COMPLETED",
                trajectory=SimpleNamespace(status="COMPLETED"),
            ),
        ],
        result_paths=[],
    )

    async def fake_submit_with_callback(client: Any, payload: dict[str, Any]) -> Any:
        del client, payload
        return task_result

    monkeypatch.setattr(
        rollout,
        "_build_task_payload",
        lambda **kwargs: {"task_id": "t1"},
    )
    monkeypatch.setattr(
        rollout,
        "_convert_task_result_to_samples",
        lambda *args, **kwargs: [
            _FakeSample(
                loss_mask=[0],
                response_length=1,
                remove_sample=True,
                metadata={"polar": {"placeholder": True}},
            ),
            _FakeSample(
                loss_mask=[1, 1],
                response_length=2,
                remove_sample=False,
                metadata={"polar": {"session_id": "s-ok"}},
            ),
        ],
    )
    monkeypatch.setattr(worker, "_submit_with_callback", fake_submit_with_callback)

    completed = _run(worker._submit_attempt(object(), pending))

    assert completed.task_id == "t1"
    assert completed.session_count == 2
    assert len(completed.samples) == 2
