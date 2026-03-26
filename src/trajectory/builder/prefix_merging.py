"""Trajectory builder that merges chained completion records into contiguous traces."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from trajectory.builder.base import BaseTrajectoryBuilder
from trajectory.builder.record_utils import build_trace_from_completion
from trajectory.models import CompletionSession, Trace, Trajectory

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
            message
            for message in messages
            if message.get("role") not in _GROUPING_IGNORED_ROLES
        ],
        ignore_patterns,
    )


def _merge_chain(chain: list[Trace]) -> Trace:
    """Merge a chain of consecutively-chained traces into one."""
    if len(chain) == 1:
        return chain[0]

    head = chain[0]
    tail = chain[-1]

    response_ids: list[int] = []
    response_messages: list[dict[str, Any]] = []
    all_logprobs: list[dict[str, Any]] = []
    all_have_logprobs = True

    for trace in chain:
        response_ids.extend(trace.response_ids)
        response_messages.extend(trace.response_messages)
        if trace.response_logprobs is not None:
            all_logprobs.extend(trace.response_logprobs)
        else:
            all_have_logprobs = False

    return Trace(
        prompt_ids=head.prompt_ids,
        prompt_messages=head.prompt_messages,
        response_ids=response_ids,
        response_messages=response_messages,
        finish_reason=tail.finish_reason,
        response_logprobs=all_logprobs if all_have_logprobs else None,
    )


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Merge chained completions into contiguous traces, splitting on compaction.

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

        chains: list[list[Trace]] = []
        waiting_chains: dict[str, deque[int]] = defaultdict(deque)

        for completion in session.completions:
            trace = build_trace_from_completion(completion)
            prompt_key = _grouping_key(
                trace.prompt_messages, self._ignore_patterns
            )
            chain_idx = self._pop_chain(prompt_key, waiting_chains)

            if chain_idx is not None:
                chains[chain_idx].append(trace)
            else:
                chain_idx = len(chains)
                chains.append([trace])

            next_key = _grouping_key(
                trace.prompt_messages + trace.response_messages,
                self._ignore_patterns,
            )
            waiting_chains[next_key].append(chain_idx)

        merged = [_merge_chain(chain) for chain in chains]

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
                "trace_count": len(merged),
            },
            traces=merged,
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
