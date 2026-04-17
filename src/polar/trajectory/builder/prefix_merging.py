"""Trajectory builder that groups chained completions by prompt prefix."""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace, Trajectory

logger = logging.getLogger(__name__)

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


def _first_user_index(messages: list[dict[str, Any]]) -> int | None:
    for i, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return i
    return None


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Group chained completions and emit one trace per chain.

    Matching is based on the **message list** rather than raw token IDs.
    Each message is normalized to ``role:content`` and the list is joined
    into a single string key for O(1) dict lookup.  Optional
    *ignore_patterns* strip harness-injected noise (e.g. cache headers,
    preamble text) before comparison so that minor differences do not
    prevent chaining.

    When *split_at_first_user* is True (default), the emitted trace splits
    the chain at the **first user turn** rather than at the final assistant
    turn: everything at or before the first user becomes the prompt, and
    everything after — including intermediate assistant/tool turns — becomes
    the training target.  This recovers multi-turn credit assignment.  When
    a chain has no user turn, or the first completion's prompt extends
    beyond the first user, the builder falls back to the final-turn-only
    trace to avoid token misalignment.

    Parameters
    ----------
    ignore_patterns:
        Regex strings (compiled with ``re.DOTALL``) applied to the
        normalized key.  Matched regions are deleted before lookup.
    split_at_first_user:
        When True, split each chain at the first user turn.  Set to False
        to reproduce the historical response-only behavior.
    """

    def __init__(
        self,
        *,
        ignore_patterns: list[str] | None = None,
        split_at_first_user: bool = True,
    ) -> None:
        self._ignore_pattern_text = list(ignore_patterns or [])
        self._ignore_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.DOTALL) for p in self._ignore_pattern_text
        ]
        self._split_at_first_user = split_at_first_user

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

        chains: list[list[CompletionRecord]] = []
        waiting_chains: dict[str, deque[int]] = defaultdict(deque)

        for completion in session.completions:
            trace = build_trace_from_completion(completion)
            prompt_key = _grouping_key(
                trace.prompt_messages, self._ignore_patterns
            )
            chain_idx = self._pop_chain(prompt_key, waiting_chains)

            if chain_idx is not None:
                chains[chain_idx].append(completion)
            else:
                chain_idx = len(chains)
                chains.append([completion])

            next_key = _grouping_key(
                trace.prompt_messages + trace.response_messages,
                self._ignore_patterns,
            )
            waiting_chains[next_key].append(chain_idx)

        final_traces = [self._finalize_chain(chain) for chain in chains]

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
                "split_at_first_user": self._split_at_first_user,
            },
            traces=final_traces,
        )

    def _finalize_chain(self, chain: list[CompletionRecord]) -> Trace:
        last_trace = build_trace_from_completion(chain[-1])
        if self._split_at_first_user:
            split_trace = self._try_split_at_first_user(chain, last_trace)
            if split_trace is not None:
                return split_trace
        return last_trace.model_copy(
            update={"response_messages": _token_aligned_response_messages(last_trace)}
        )

    def _try_split_at_first_user(
        self,
        chain: list[CompletionRecord],
        last_trace: Trace,
    ) -> Trace | None:
        first_trace = build_trace_from_completion(chain[0])
        prefix_messages = first_trace.prompt_messages

        # Only split when the first completion's prompt ends at the first user
        # turn — the standard agent flow.  Otherwise token-level alignment
        # between C1.prompt_ids and the full conversation is not guaranteed.
        user_idx = _first_user_index(prefix_messages)
        if user_idx is None or user_idx != len(prefix_messages) - 1:
            return None

        prompt_ids = list(first_trace.prompt_ids)
        full_ids = list(last_trace.prompt_ids) + list(last_trace.response_ids)
        if not prompt_ids or len(prompt_ids) > len(full_ids):
            return None
        if full_ids[: len(prompt_ids)] != prompt_ids:
            logger.debug(
                "prefix_merging split skipped: first completion prompt_ids is not a "
                "prefix of the final completion's tokenized conversation"
            )
            return None

        response_ids = full_ids[len(prompt_ids):]
        response_messages = (
            list(last_trace.prompt_messages[len(prefix_messages):])
            + list(last_trace.response_messages)
        )
        response_logprobs = self._collect_response_logprobs(
            chain=chain,
            prompt_len=len(prompt_ids),
            response_ids=response_ids,
            full_ids=full_ids,
        )

        return Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_messages=list(prefix_messages),
            response_messages=response_messages,
            finish_reason=last_trace.finish_reason,
            response_logprobs=response_logprobs,
        )

    @staticmethod
    def _collect_response_logprobs(
        *,
        chain: list[CompletionRecord],
        prompt_len: int,
        response_ids: list[int],
        full_ids: list[int],
    ) -> list[dict[str, Any]] | None:
        """Stitch each completion's logprobs into the chain-wide response slot.

        Each completion in the chain contributes logprobs for its own assistant
        tokens at the offset where those tokens land in *full_ids*.  Interstitial
        positions (chat-template markers, tool results, intermediate user turns)
        are zero-filled so TIS can correct the off-policy ratio trainer-side.
        """
        slots: list[dict[str, Any] | None] = [None] * len(response_ids)
        any_real = False

        for completion in chain:
            comp_trace = build_trace_from_completion(completion)
            if not comp_trace.response_logprobs:
                continue
            start = len(comp_trace.prompt_ids) - prompt_len
            end = start + len(comp_trace.response_ids)
            if start < 0 or end > len(response_ids):
                continue
            if full_ids[len(comp_trace.prompt_ids):len(comp_trace.prompt_ids) + len(comp_trace.response_ids)] != list(comp_trace.response_ids):
                continue
            for j, entry in enumerate(comp_trace.response_logprobs):
                if isinstance(entry, dict):
                    slots[start + j] = entry
                    any_real = True

        if not any_real:
            return None

        return [
            slot if slot is not None else {"token_id": response_ids[i], "logprob": 0.0}
            for i, slot in enumerate(slots)
        ]

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
