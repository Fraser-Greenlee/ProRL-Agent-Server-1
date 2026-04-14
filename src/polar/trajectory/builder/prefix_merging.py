"""Trajectory builder that groups chained completions by prompt prefix."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionSession, Trace, Trajectory

_GROUPING_IGNORED_ROLES = frozenset({"tool"})


def _flatten_message_content(content: Any) -> str:
    """Extract text from a message content field (string or content-part array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _normalize_messages(
    messages: list[dict[str, Any]],
    ignore_patterns: list[re.Pattern[str]],
) -> str:
    """Flatten a message list into a deterministic key string.

    Format: ``role:content<SEP>role:content<SEP>...``

    *ignore_patterns* are applied to the final string so that matched regions
    (e.g. harness-injected cache headers) are stripped before comparison.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            content = ""
        else:
            content = _flatten_message_content(msg.get("content"))
        parts.append(f"{role}:{content}")
    key = "<SEP>".join(parts)
    for pattern in ignore_patterns:
        key = pattern.sub("", key)
    return key


def _grouping_key(
    messages: list[dict[str, Any]],
    ignore_patterns: list[re.Pattern[str]],
) -> str:
    """Normalize the structural conversation context used for chaining.

    Tool-result messages are omitted because they are harness artifacts that
    appear between assistant turns in the next request prompt.
    """
    return _normalize_messages(
        [
            expanded_message
            for message in messages
            for expanded_message in _expand_messages_for_grouping(message)
            if not _is_grouping_noise_message(expanded_message)
        ],
        ignore_patterns,
    )


def _expand_messages_for_grouping(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role != "assistant" or not message.get("tool_calls"):
        return [message]

    expanded: list[dict[str, Any]] = []
    content = message.get("content")
    if content not in (None, "", []):
        expanded.append(
            {
                "role": role,
                "content": content,
            }
        )
    expanded.append(
        {
            "role": role,
            "content": None,
            "tool_calls": message.get("tool_calls"),
        }
    )
    return expanded


def _is_grouping_noise_message(message: dict[str, Any]) -> bool:
    role = message.get("role")
    if role in _GROUPING_IGNORED_ROLES:
        return True
    if role == "assistant" and message.get("tool_calls"):
        return False
    content = _flatten_message_content(message.get("content")).strip()
    if role == "assistant" and content.startswith("<think>"):
        return True
    if role == "assistant" and not content and not message.get("tool_calls"):
        return True
    return False


def _token_aligned_response_messages(trace: Trace) -> list[dict[str, Any]]:
    if not trace.response_logprobs:
        return trace.response_messages
    tokens = [
        str(item.get("token", ""))
        for item in trace.response_logprobs
        if isinstance(item, dict)
    ]
    exact_text = "".join(tokens)
    if not exact_text:
        return trace.response_messages
    return [{"role": "assistant", "content": exact_text}]


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Group chained completions and emit the final trace from each group.

    Matching is based on the **message list** rather than raw token IDs.
    Each message is normalized to ``role:content`` and the list is joined
    into a single string key for O(1) dict lookup.  Optional
    *ignore_patterns* strip harness-injected noise (e.g. cache headers,
    preamble text) before comparison so that minor differences do not
    prevent chaining.

    Parameters
    ----------
    ignore_patterns:
        Regex strings (compiled with ``re.DOTALL``) applied to the
        normalized key.  Matched regions are deleted before lookup.
    """

    def __init__(
        self,
        *,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self._ignore_pattern_text = list(ignore_patterns or [])
        self._ignore_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.DOTALL) for p in self._ignore_pattern_text
        ]

    async def build(self, session: CompletionSession) -> Trajectory:
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "prefix_merging",
                    "session_id": session.session_id,
                    "record_count": 0,
                },
                traces=[],
                error="no completions",
            )

        chains: list[Trace] = []
        waiting_chains: dict[str, deque[int]] = defaultdict(deque)

        for completion in session.completions:
            trace = build_trace_from_completion(completion)
            prompt_key = _grouping_key(
                trace.prompt_messages, self._ignore_patterns
            )
            chain_idx = self._pop_chain(prompt_key, waiting_chains)

            if chain_idx is not None:
                chains[chain_idx] = trace
            else:
                chain_idx = len(chains)
                chains.append(trace)

            next_key = _grouping_key(
                trace.prompt_messages + trace.response_messages,
                self._ignore_patterns,
            )
            waiting_chains[next_key].append(chain_idx)

        final_traces = [
            trace.model_copy(update={"response_messages": _token_aligned_response_messages(trace)})
            for trace in chains
        ]

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "prefix_merging",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(session.completions),
                "trace_count": len(chains),
            },
            traces=final_traces,
        )

    @staticmethod
    def _pop_chain(
        prompt_key: str,
        waiting_chains: dict[str, deque[int]],
    ) -> int | None:
        """Pop and return the chain index whose expected-next-prompt matches *prompt_key*."""
        queue = waiting_chains.get(prompt_key)
        if queue:
            chain_idx = queue.popleft()
            if not queue:
                waiting_chains.pop(prompt_key, None)
            return chain_idx
        return None
