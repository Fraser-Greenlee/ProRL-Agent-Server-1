"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.  All
samples produced from the same session share the same ``Sample.index``
(the trajectory's position within the group) so the reward post-processor
can treat them as one trajectory.  Builders own trace curation — the
adapter does not filter; traces that lack training tokens are dropped and
represented as fully masked samples so callers can keep the rest of the
group trainable.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from slime_bridge._messages import flatten_content, messages_to_text

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)


class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
    tokenizer_name_or_path: str | None = None,
    add_generation_prompt: bool = True,
) -> list[Any]:
    """Convert one Polar session result into Slime samples — one per trace.

    Every usable trace becomes an independent Sample sharing the same
    ``(group_index, index)`` key. Slime's reward post-processor collapses
    them back into a single trajectory for advantage normalization, but
    every trace contributes its own assistant-generated tokens to the
    gradient. 

    Traces with empty tokens or exceeding ``max_tokens`` are dropped
    (logged). If *all* traces are dropped we emit a single zero-gradient
    placeholder so Slime's flattener doesn't crash on an empty list and
    the rest of the group can still train.
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
            tokenizer_name_or_path=tokenizer_name_or_path,
            add_generation_prompt=add_generation_prompt,
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
    tokenizer_name_or_path: str | None = None,
    add_generation_prompt: bool = True,
) -> Any | None:
    prompt_ids = _resolve_prompt_ids(
        trace,
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_generation_prompt=add_generation_prompt,
        session_id=result.session_id,
        trace_index=trace_index,
    )
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

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(trace)

    trainable = status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
    loss_mask = _loss_mask_from_logprobs(
        trace,
        len(response_ids),
        require_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
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
    polar_metadata.update(_scheduler_metadata(result, trace))

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
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    Polar metrics.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
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
    polar_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). slime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _sample_status(Sample: Any, result: "SessionResult", trace: "Trace") -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    values: list[float] = []
    for pos, (entry, mask_value) in enumerate(zip(logprobs, loss_mask, strict=True)):
        if not isinstance(entry, dict):
            if mask_value:
                raise RolloutLogprobError(
                    f"Session {session_id} trace {trace_index}: logprob entry {pos} "
                    "is not a mapping"
                )
            values.append(0.0)
            continue
        if mask_value and "logprob" not in entry:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: trainable token {pos} "
                "is missing logprob"
            )
        values.append(float(entry.get("logprob", 0.0)))
    return values


def _loss_mask_from_logprobs(
    trace: "Trace",
    response_len: int,
    *,
    require_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Build a per-token loss mask from the trace's response_logprobs.

    Prefix-merging builders produce a mixed token stream: raw assistant
    tokens interleaved with canonical interstitials (tool responses,
    chat-template glue).  Only the former should contribute to training.

    The builder marks them distinctly in ``response_logprobs``: real
    server-returned entries either carry the internal ``"_polar_trainable"``
    marker or include a ``"token"`` (string) field; interstitial slots are
    synthesized as ``{"token_id": ..., "logprob": 0.0}`` with neither
    marker.  We do not key this off ``logprob == 0.0`` because legitimate
    high-confidence tokens can also hit logprob 0.

    Trainable traces must carry one logprob entry per response token.
    """
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_logprobs:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing response_logprobs"
            )
        return [1] * response_len
    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: response_logprobs length "
            f"{len(logprobs)} != response length {response_len}"
        )
    mask = [1 if _is_trainable_logprob_entry(entry) else 0 for entry in logprobs]
    return mask


def _is_trainable_logprob_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if bool(entry.get("_polar_trainable")):
        return True
    return "token" in entry


def _response_ids_from_logprobs(trace: "Trace") -> list[int]:
    if not trace.response_logprobs:
        return []
    return [
        int(item["token_id"])
        for item in trace.response_logprobs
        if isinstance(item, dict) and item.get("token_id") is not None
    ]


def _resolve_prompt_ids(
    trace: "Trace",
    *,
    tokenizer_name_or_path: str | None,
    add_generation_prompt: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    if trace.prompt_ids:
        return list(trace.prompt_ids)
    if not tokenizer_name_or_path or not trace.prompt_messages:
        return []

    try:
        tokenizer = _load_tokenizer(tokenizer_name_or_path)
        encoded = tokenizer.apply_chat_template(
            _normalize_chat_template_messages(trace.prompt_messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        token_ids = _coerce_token_ids(encoded)
        if token_ids:
            return token_ids
    except Exception as exc:
        logger.warning(
            "Session %s trace %d: failed to tokenize prompt_messages via chat template: %s",
            session_id, trace_index, exc,
        )

    try:
        tokenizer = _load_tokenizer(tokenizer_name_or_path)
        rendered_prompt = messages_to_text(trace.prompt_messages)
        return _coerce_token_ids(
            tokenizer.encode(rendered_prompt, add_special_tokens=False)
        )
    except Exception as exc:
        logger.warning(
            "Session %s trace %d: failed to tokenize fallback prompt text: %s",
            session_id, trace_index, exc,
        )
        return []


def _normalize_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = flatten_content(message.get("content"))
        if role == "tool":
            tool_name = str(message.get("name") or message.get("tool_call_id") or "tool")
            content = f"[tool result: {tool_name}]\n{content}".strip()
            role = "user"
        elif role not in {"system", "user", "assistant"}:
            role = "user"

        tool_calls = _tool_calls_to_text(message.get("tool_calls"))
        if tool_calls:
            content = f"{content}\n{tool_calls}".strip()

        normalized.append({"role": role, "content": content})
    return normalized


def _tool_calls_to_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name") or call.get("name") or call.get("id") or "tool"
            arguments = function.get("arguments")
        else:
            name = call.get("name") or call.get("id") or "tool"
            arguments = call.get("arguments")
        text = f"[tool call: {name}]"
        if arguments not in (None, ""):
            text = f"{text} {arguments}"
        parts.append(text)
    return "\n".join(parts)


def _coerce_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    elif hasattr(encoded, "data") and isinstance(getattr(encoded, "data"), dict):
        encoded = encoded.data.get("input_ids", [])
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if (
        isinstance(encoded, list)
        and encoded
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = list(encoded[0])
    if not isinstance(encoded, list):
        return []
    return [int(token_id) for token_id in encoded if isinstance(token_id, int)]


@lru_cache(maxsize=4)
def _load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if Path(name_or_path).exists():
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
