"""Chain builder with raw-response + canonical-interstitial token stream.

Motivation
----------
An earlier revision detected chains at the **message** level but split the
merged trace at the **token** level by requiring ``first.prompt_ids`` to be
a byte-exact prefix of ``last.prompt_ids + last.response_ids``.  Any drift
across requests (context-sensitive BPE merges, template variants) tripped
that global check and the whole chain silently collapsed to the last
completion alone.

A follow-up revision appended each completion's ``response_ids`` and did a
*local* prefix check between adjacent completions — but the same BPE drift
problem resurfaced on every turn (``tokenize(decode(raw_response)) !=
raw_response`` at multi-line JSON / special chars), collapsing chains to
2–3 turns in practice.

Current approach (raw + canonical interstitial)
-----------------------------------------------

* The **assistant body** comes from the raw sampled ids (``C_i.response_ids``).
  These are what the model actually emitted, so their logprobs are real.
  We never decode→re-encode them, so BPE drift cannot bite.
* The **interstitials** (tool results, intermediate user turns, chat-template
  glue like ``<|im_end|>\\n<|im_start|>...<|im_end|>\\n<|im_start|>assistant``)
  come from ``C_{i+1}.prompt_ids`` — the server's canonical tokenization.

The critical insight is that the prefix check is **canonical-vs-canonical**
(``C_{i+1}.prompt_ids[:len(C_i.prompt_ids)] == C_i.prompt_ids``) rather than
raw-vs-canonical, and therefore passes reliably.

To split the canonical tail into "canonical C_i response" (discard) vs
"interstitial" (keep), we scan for the first end-of-turn token
(``<|im_end|>`` on Qwen-family, configurable).  Everything past that
marker is canonical chat-template glue + tool messages and is appended
to the stream.  If the raw response already ended with the end-of-turn
token (the usual case when ``finish_reason`` ∈ {stop, tool_calls}), we
skip the duplicate; otherwise (truncation) we prepend it.

Stream construction
-------------------
    stream = list(C_1.prompt_ids)
    stream += C_1.response_ids           # raw
    for C_{i+1} in chain[1:]:
        if C_{i+1}.prompt_ids[:len(C_i.prompt_ids)] != C_i.prompt_ids:
            break                         # message prefix diverged
        tail = C_{i+1}.prompt_ids[len(C_i.prompt_ids):]
        K = tail.index(EOT)               # first end-of-turn token
        interstitial = tail[K:] or tail[K+1:]   # see _slice_interstitial
        stream += interstitial + list(C_{i+1}.response_ids)

Interstitial tokens get ``None`` logprob slots — the adapter must mask
them out of loss.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace, Trajectory

logger = logging.getLogger(__name__)

# finish_reasons where the model emitted the natural end-of-turn token itself.
_NATURAL_STOP_REASONS = frozenset({"stop", "tool_calls", "stop_sequence"})

_GROUPING_IGNORED_ROLES = frozenset({"tool"})


# ---------------------------------------------------------------------------
# Message-level grouping helpers — used to detect which completions belong
# to the same agentic chain (C_{i+1}'s prompt == C_i's prompt + response).
# ---------------------------------------------------------------------------


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


def _expand_messages_for_grouping(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role != "assistant" or not message.get("tool_calls"):
        return [message]

    expanded: list[dict[str, Any]] = []
    content = message.get("content")
    if content not in (None, "", []):
        expanded.append({"role": role, "content": content})
    expanded.append(
        {"role": role, "content": None, "tool_calls": message.get("tool_calls")}
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


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Rebuild a chain's merged token stream using raw + canonical-interstitial.

    Parameters
    ----------
    ignore_patterns:
        Regex strings (DOTALL) applied to the normalized message key
        during chain detection.  Matched regions are stripped before
        comparison so harness-injected noise (cache headers, etc.) does
        not prevent chaining.  Does not affect token-level
        reconstruction.
    end_of_turn_token_id:
        Explicit end-of-turn (EOT) token id used to locate the
        canonical-tail split between the prior assistant body and the
        interstitial.  When None (default), the builder auto-detects it
        from the last token of the first completion with a natural stop
        reason.  For Qwen / ChatML templates this is the
        ``<|im_end|>`` token id.
    """

    def __init__(
        self,
        *,
        ignore_patterns: list[str] | None = None,
        end_of_turn_token_id: int | None = None,
    ) -> None:
        self._ignore_pattern_text = list(ignore_patterns or [])
        self._ignore_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.DOTALL) for p in self._ignore_pattern_text
        ]
        self._configured_eot_id = end_of_turn_token_id

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
            prompt_key = _grouping_key(trace.prompt_messages, self._ignore_patterns)
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

        stats: dict[str, int] = {
            "chains_total": len(chains),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "completions_total": len(session.completions),
            "completions_merged": 0,
        }
        final_traces = [self._finalize_chain(chain, stats) for chain in chains]

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
                "reconstruction_stats": stats,
            },
            traces=final_traces,
        )

    # ------------------------------------------------------------------
    # Chain finalization
    # ------------------------------------------------------------------

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        stats: dict[str, int],
    ) -> Trace:
        # Everything in C_1.prompt_ids is the non-trainable
        # prompt; C_1.response_ids plus every subsequent raw response +
        # canonical interstitial becomes the trainable response.  No role-shape
        # constraint on the initial conversation — a harness preamble like
        # codex's [system, user, user, assistant, tool, ...] is treated as
        # static context.
        first_trace = build_trace_from_completion(chain[0])
        eot_id = self._resolve_eot_id(chain)

        prompt_ids = list(first_trace.prompt_ids)
        stream_ids: list[int] = list(prompt_ids)
        response_slots: list[dict[str, Any] | None] = []
        response_messages: list[dict[str, Any]] = []

        # Track the canonical prompt_ids of the most recently merged
        # completion — used for the canonical-vs-canonical prefix check.
        prev_prompt_ids: list[int] = list(first_trace.prompt_ids)
        prev_raw_response: list[int] = list(first_trace.response_ids)

        # Running count of messages consumed = prompt_messages + all response_messages emitted.
        msg_acc = len(first_trace.prompt_messages)

        self._append_response_tokens(first_trace, stream_ids, response_slots)
        response_messages.extend(deepcopy(m) for m in first_trace.response_messages)
        msg_acc += len(first_trace.response_messages)
        kept = 1

        for i in range(1, len(chain)):
            Ci_trace = build_trace_from_completion(chain[i])
            Ci_prompt_ids = list(Ci_trace.prompt_ids)

            # Canonical-vs-canonical prefix check: both sides are server-side
            # tokenizations of the same message prefix — matches reliably
            # unless the harness rewrote prior messages.
            if (
                len(Ci_prompt_ids) < len(prev_prompt_ids)
                or Ci_prompt_ids[: len(prev_prompt_ids)] != prev_prompt_ids
            ):
                logger.debug(
                    "prefix_merging: canonical prefix break at step %d/%d",
                    i,
                    len(chain),
                )
                break

            # canonical_tail = canonical tokens for [prev assistant msg + new interstitials].
            canonical_tail = Ci_prompt_ids[len(prev_prompt_ids):]
            interstitial = self._slice_interstitial(
                canonical_tail=canonical_tail,
                prev_raw_response=prev_raw_response,
                eot_id=eot_id,
            )
            if interstitial is None:
                logger.debug(
                    "prefix_merging: interstitial split failed at step %d/%d "
                    "(eot_id=%r, tail_len=%d)",
                    i,
                    len(chain),
                    eot_id,
                    len(canonical_tail),
                )
                break

            if interstitial:
                stream_ids.extend(interstitial)
                response_slots.extend([None] * len(interstitial))

            # Message-level interstitial bookkeeping.
            if len(Ci_trace.prompt_messages) > msg_acc:
                interstitial_msgs = Ci_trace.prompt_messages[msg_acc:]
                response_messages.extend(deepcopy(m) for m in interstitial_msgs)
                msg_acc += len(interstitial_msgs)

            self._append_response_tokens(Ci_trace, stream_ids, response_slots)
            response_messages.extend(deepcopy(m) for m in Ci_trace.response_messages)
            msg_acc += len(Ci_trace.response_messages)

            prev_prompt_ids = Ci_prompt_ids
            prev_raw_response = list(Ci_trace.response_ids)
            kept += 1

        stats["completions_merged"] += kept
        if kept == len(chain):
            stats["chains_reconstructed_full"] += 1
        else:
            stats["chains_reconstructed_truncated"] += 1

        response_ids = stream_ids[len(prompt_ids):]
        response_logprobs = self._finalize_logprobs(response_slots, response_ids)
        last_kept_trace = build_trace_from_completion(chain[kept - 1])

        return Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_messages=[deepcopy(m) for m in first_trace.prompt_messages],
            response_messages=response_messages,
            finish_reason=last_kept_trace.finish_reason,
            response_logprobs=response_logprobs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_eot_id(self, chain: list[CompletionRecord]) -> int | None:
        """Return configured EOT id, else auto-detect from the chain.

        Auto-detection uses the last token of the first completion whose
        ``finish_reason`` indicates the model emitted the natural stop
        marker itself (stop / tool_calls / stop_sequence).
        """
        if self._configured_eot_id is not None:
            return self._configured_eot_id
        for completion in chain:
            trace = build_trace_from_completion(completion)
            if (
                trace.finish_reason in _NATURAL_STOP_REASONS
                and trace.response_ids
            ):
                return trace.response_ids[-1]
        return None

    @staticmethod
    def _slice_interstitial(
        *,
        canonical_tail: list[int],
        prev_raw_response: list[int],
        eot_id: int | None,
    ) -> list[int] | None:
        """Extract the canonical interstitial from C_{i+1}'s prompt tail.

        ``canonical_tail`` = canonical tokens for [prev assistant msg +
        harness-inserted messages + generation-prompt glue].  The first
        occurrence of ``eot_id`` marks the end of the prev assistant
        body; everything after is interstitial.

        If ``prev_raw_response`` already ends with ``eot_id`` (natural
        stop / tool_calls), skip it in the canonical tail to avoid
        duplication; otherwise (truncation) include it so the stream
        still closes the assistant turn.

        Returns None if ``eot_id`` is unknown or not present — caller
        should treat this as a break.
        """
        if eot_id is None:
            return None
        try:
            k = canonical_tail.index(eot_id)
        except ValueError:
            return None
        if prev_raw_response and prev_raw_response[-1] == eot_id:
            return canonical_tail[k + 1 :]
        return canonical_tail[k:]

    @staticmethod
    def _append_response_tokens(
        trace: Trace,
        stream_ids: list[int],
        response_slots: list[dict[str, Any] | None],
    ) -> None:
        """Append a completion's response_ids and parallel logprob slots."""
        response_ids = list(trace.response_ids)
        stream_ids.extend(response_ids)
        logprobs = trace.response_logprobs or []
        for pos in range(len(response_ids)):
            entry = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(deepcopy(entry) if isinstance(entry, dict) else None)

    @staticmethod
    def _finalize_logprobs(
        slots: list[dict[str, Any] | None],
        response_ids: list[int],
    ) -> list[dict[str, Any]] | None:
        if not any(slot is not None for slot in slots):
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
        queue = waiting_chains.get(prompt_key)
        if queue:
            chain_idx = queue.popleft()
            if not queue:
                waiting_chains.pop(prompt_key, None)
            return chain_idx
        return None
