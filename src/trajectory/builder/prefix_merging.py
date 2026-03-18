"""Trajectory builder that merges chained completion records into contiguous traces."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from trajectory.builder.base import BaseTrajectoryBuilder
from trajectory.builder.record_utils import build_trace_from_completion
from trajectory.models import CompletionSession, Trace, Trajectory


def _edit_distance_within_budget(
    a: tuple[int, ...], b: tuple[int, ...], budget: int
) -> int | None:
    """Levenshtein distance if <= *budget*, else ``None``.

    Uses banded DP so the cost is O(n * budget) rather than O(n * m).
    """
    n, m = len(a), len(b)
    if abs(n - m) > budget:
        return None
    if n == 0:
        return m if m <= budget else None
    if m == 0:
        return n if n <= budget else None

    if n > m:
        a, b = b, a
        n, m = m, n

    INF = budget + 1
    prev = [INF] * (m + 1)
    for j in range(min(m, budget) + 1):
        prev[j] = j

    for i in range(1, n + 1):
        curr = [INF] * (m + 1)
        if i <= budget:
            curr[0] = i

        lo = max(1, i - budget)
        hi = min(m, i + budget)

        for j in range(lo, hi + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])

        prev = curr

    return prev[m] if prev[m] <= budget else None


def _sequence_similarity(
    a: tuple[int, ...], b: tuple[int, ...], *, min_ratio: float = 0.0
) -> float:
    """Token-level similarity: ``1 - edit_distance(a, b) / max(len(a), len(b))``.

    *min_ratio* enables early termination: when the similarity provably cannot
    reach *min_ratio* the function returns ``0.0`` without a full DP pass.
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    budget = int(max_len * (1.0 - min_ratio))
    dist = _edit_distance_within_budget(a, b, budget)
    if dist is None:
        return 0.0
    return 1.0 - dist / max_len


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

    Parameters
    ----------
    match_tolerance:
        Minimum token-level similarity (``1 − edit_distance / max_len``) for
        two consecutive turns to be considered part of the same chain.
        ``1.0`` (default) requires an exact id-by-id match.  Lower values
        (e.g. ``0.99``) tolerate small request-id or token tweaks between
        turns.
    """

    def __init__(self, *, match_tolerance: float = 1.0) -> None:
        if not 0.0 < match_tolerance <= 1.0:
            raise ValueError("match_tolerance must be in (0, 1]")
        self._match_tolerance = match_tolerance

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
        waiting_chains: dict[tuple[int, ...], deque[int]] = defaultdict(deque)
        exact = self._match_tolerance >= 1.0

        for completion in session.completions:
            trace = build_trace_from_completion(completion)
            prompt_key = tuple(trace.prompt_ids)
            chain_idx = self._find_chain(prompt_key, waiting_chains, exact)

            if chain_idx is not None:
                chains[chain_idx].append(trace)
            else:
                chain_idx = len(chains)
                chains.append([trace])

            next_prompt_key = tuple(trace.prompt_ids + trace.response_ids)
            waiting_chains[next_prompt_key].append(chain_idx)

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
                "match_tolerance": self._match_tolerance,
            },
            traces=merged,
        )

    def _find_chain(
        self,
        prompt_key: tuple[int, ...],
        waiting_chains: dict[tuple[int, ...], deque[int]],
        exact: bool,
    ) -> int | None:
        """Pop and return the chain index whose expected-next-prompt matches *prompt_key*."""
        if exact:
            queue = waiting_chains.get(prompt_key)
            if queue:
                chain_idx = queue.popleft()
                if not queue:
                    waiting_chains.pop(prompt_key, None)
                return chain_idx
            return None

        best_key: tuple[int, ...] | None = None
        best_sim = 0.0

        for key in waiting_chains:
            sim = _sequence_similarity(prompt_key, key, min_ratio=self._match_tolerance)
            if sim >= self._match_tolerance and sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None:
            queue = waiting_chains[best_key]
            chain_idx = queue.popleft()
            if not queue:
                del waiting_chains[best_key]
            return chain_idx

        return None
