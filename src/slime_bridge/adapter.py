"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.  All
samples produced from the same session share the same ``Sample.index``
(the trajectory's position within the group) so the reward post-processor
can treat them as one trajectory.  Builders own trace curation — the
adapter does not filter; traces that lack training tokens are dropped so
callers never smuggle placeholder tokens into the training batch.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, TYPE_CHECKING

from slime_bridge._messages import messages_to_text

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one Polar session result into Slime samples — one per trace.

    Every usable trace becomes an independent Sample sharing the same
    ``(group_index, index)`` key. Slime's reward post-processor collapses
    them back into a single trajectory for advantage normalization, but
    every trace contributes its own assistant-generated tokens to the
    gradient. 

    Traces with empty tokens or exceeding ``max_tokens`` are dropped
    (logged). If *all* traces are dropped we emit a single zero-gradient
    placeholder so Slime's flattener doesn't crash on an empty list.
    """
    Sample = _load_sample_type()
    traces = result.trajectory.traces
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=trajectory_index,
            reward_key=reward_key,
            max_tokens=max_tokens,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id, len(traces), max_tokens,
    )
    return [_build_dummy_sample(
        Sample=Sample,
        result=result,
        group_index=group_index,
        index=trajectory_index,
        reward_key=reward_key,
    )]


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int,
    reward_key: str,
    max_tokens: int | None = None,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids) or _response_ids_from_logprobs(trace)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    total_len = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and total_len > max_tokens:
        logger.warning(
            "Dropping trace %d from session %s: total_len=%d > max_tokens=%d",
            trace_index, result.session_id, total_len, max_tokens,
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    response_log_probs = _extract_rollout_log_probs(trace)
    if not response_log_probs:
        response_log_probs = [0.0] * len(response_ids)

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(trace)

    loss_mask = [1] * len(response_ids)
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        # Preserved for the longest-trace wandb artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(response_messages),
        },
    }

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        metadata={"polar": polar_metadata},
    )


def _build_dummy_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    group_index: int,
    index: int,
    reward_key: str,
) -> Any:
    """Minimal near-zero-gradient placeholder — keeps per-session sample
    count at 1 so slime's _get_rollout_data doesn't crash on an empty
    flattened list. ``loss_mask=[1]`` ensures the global mask sum stays
    non-zero even if every session in a batch fails; distributed_masked_whiten
    crashes on sum=0. The single-token <pad→pad> prediction contributes a
    negligible, benign gradient.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        reward={reward_key: 0.0},
        loss_mask=[1],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). slime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _sample_status(Sample: Any, result: "SessionResult", trace: "Trace") -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(trace: "Trace") -> list[float]:
    if not trace.response_logprobs:
        return []
    return [
        float(item.get("logprob", 0.0))
        for item in trace.response_logprobs
        if isinstance(item, dict)
    ]


def _response_ids_from_logprobs(trace: "Trace") -> list[int]:
    if not trace.response_logprobs:
        return []
    return [
        int(item["token_id"])
        for item in trace.response_logprobs
        if isinstance(item, dict) and item.get("token_id") is not None
    ]


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
