"""Shared completion-session and trajectory schemas used by rollout nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Strategy / evaluator specs
# ---------------------------------------------------------------------------


class StrategySpec(BaseModel):
    """Identifies a builder or evaluator strategy with optional per-request config."""

    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluatorSpec(BaseModel):
    """Evaluator configuration for a rollout session."""

    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    refresh_runtime: bool = False


# ---------------------------------------------------------------------------
# Evaluator result
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Evaluator output — the gateway merges rewards into the trajectory."""

    outcome_reward: float | None = None
    trace_rewards: list[float | None] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Completion and trajectory models
# ---------------------------------------------------------------------------


class CompletionRecord(BaseModel):
    """One normalized upstream completion payload."""

    model_config = ConfigDict(extra="allow")

    completion_id: str
    timestamp: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    original_request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)


class CompletionSession(BaseModel):
    """Builder-facing session payload with metadata and ordered completions."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    created_at: str | None = None
    completion_count: int = 0
    task_id: str | None = None
    model_requested: str | None = None
    model_used: str | None = None
    api_type: str | None = None
    completions: list[CompletionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_completions(self) -> "CompletionSession":
        self.completions = sorted(
            self.completions,
            key=lambda completion: (
                str(completion.timestamp or ""),
                completion.completion_id,
            ),
        )
        return self


class Trace(BaseModel):
    """One reconstructed completion interaction."""

    prompt_ids: list[int] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    response_messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    response_logprobs: list[dict[str, Any]] | None = None
    reward: float | None = None
    advantage: float | None = None


class Trajectory(BaseModel):
    """Structured trajectory reconstructed from session completion records."""

    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    traces: list[Trace] = Field(default_factory=list)
    error: str | None = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        allowed = {"COMPLETED", "TIMEOUT", "ERROR"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value
