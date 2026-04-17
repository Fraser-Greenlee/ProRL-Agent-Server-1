"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.
Builders own trace curation — the adapter does not filter. Traces that
lack training tokens (empty prompt_ids or response_ids) are dropped so
callers never smuggle placeholder tokens into the training batch.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import itertools
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
    reward_key: str = "score",
    index: int | None = None,
    next_index: Callable[[], int] | None = None,
) -> list[Any]:
    """Convert one Polar session result into Slime samples (one per trace)."""
    Sample = _load_sample_type()
    traces = list(result.trajectory.traces)

    if next_index is None:
        counter = itertools.count(0 if index is None else index)
        next_index = counter.__next__

    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample_index = index if (index is not None and trace_index == 0) else next_index()
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=sample_index,
            reward_key=reward_key,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int | None,
    reward_key: str,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids) or _response_ids_from_logprobs(trace)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    response_log_probs = _extract_rollout_log_probs(trace)
    if not response_log_probs:
        response_log_probs = [0.0] * len(response_ids)

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(result, trace)

    loss_mask = [1] * len(response_ids)
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_error": result.error,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
    }
    if trace.advantage is not None:
        polar_metadata["advantage"] = float(trace.advantage)

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
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
    )


def _reward_value(result: "SessionResult", trace: "Trace") -> float:
    if trace.reward is not None:
        return float(trace.reward)

    evaluation = result.trajectory.metadata.get("evaluation", {})
    if isinstance(evaluation, dict) and evaluation.get("outcome_reward") is not None:
        return float(evaluation["outcome_reward"])
    return 0.0


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
