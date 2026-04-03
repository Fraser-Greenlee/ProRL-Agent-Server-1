from __future__ import annotations

import pytest

from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def _completion(
    completion_id: str,
    timestamp: str,
    prompt_messages: list[dict],
    response_message: dict,
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
) -> CompletionRecord:
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=timestamp,
        request={"messages": prompt_messages},
        response={
            "prompt_token_ids": prompt_ids or [],
            "choices": [
                {
                    "message": response_message,
                    "finish_reason": "stop",
                    "token_ids": response_ids or [],
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_prefix_merging_uses_tail_trace_for_chained_records() -> None:
    session = CompletionSession(
        session_id="session-1",
        task_id="task-1",
        model_requested="model-a",
        model_used="model-a",
        api_type="openai_chat",
        completions=[
            _completion(
                "c1",
                "2026-03-30T00:00:01Z",
                [{"role": "user", "content": "Solve 2+2"}],
                {"role": "assistant", "content": "4"},
                prompt_ids=[1, 2],
                response_ids=[10],
            ),
            _completion(
                "c2",
                "2026-03-30T00:00:02Z",
                [
                    {"role": "user", "content": "Solve 2+2"},
                    {"role": "assistant", "content": "4"},
                ],
                {"role": "assistant", "content": "Anything else?"},
                prompt_ids=[3, 4, 5],
                response_ids=[20],
            ),
        ],
    )

    trajectory = await PrefixMergingBuilder().build(session)

    assert trajectory.status == "COMPLETED"
    assert trajectory.metadata["record_count"] == 2
    assert trajectory.metadata["trace_count"] == 1
    assert [trace.model_dump() for trace in trajectory.traces] == [
        {
            "prompt_ids": [3, 4, 5],
            "response_ids": [20],
            "prompt_messages": [
                {"role": "user", "content": "Solve 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            "response_messages": [
                {"role": "assistant", "content": "Anything else?"}
            ],
            "finish_reason": "stop",
            "response_logprobs": None,
            "reward": None,
        }
    ]


@pytest.mark.asyncio
async def test_prefix_merging_preserves_tail_prompt_with_tool_result() -> None:
    session = CompletionSession(
        session_id="session-2",
        completions=[
            _completion(
                "c1",
                "2026-03-30T00:00:01Z",
                [{"role": "user", "content": "Use a tool"}],
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"expression\":\"2+2\"}",
                            },
                        }
                    ],
                },
                prompt_ids=[1],
                response_ids=[11],
            ),
            _completion(
                "c2",
                "2026-03-30T00:00:02Z",
                [
                    {"role": "user", "content": "Use a tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": "{\"expression\":\"2+2\"}",
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "4"},
                ],
                {"role": "assistant", "content": "The answer is 4."},
                prompt_ids=[2, 3, 4],
                response_ids=[21],
            ),
        ],
    )

    trajectory = await PrefixMergingBuilder().build(session)

    assert trajectory.metadata["trace_count"] == 1
    assert [trace.model_dump() for trace in trajectory.traces] == [
        {
            "prompt_ids": [2, 3, 4],
            "response_ids": [21],
            "prompt_messages": [
                {"role": "user", "content": "Use a tool"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"expression\":\"2+2\"}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "4"},
            ],
            "response_messages": [
                {"role": "assistant", "content": "The answer is 4."}
            ],
            "finish_reason": "stop",
            "response_logprobs": None,
            "reward": None,
        }
    ]
