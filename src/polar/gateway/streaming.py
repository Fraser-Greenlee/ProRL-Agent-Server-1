"""Stream accumulator — collects chunks for storage while forwarding them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StreamAccumulator:
    """Accumulates streaming chunks for later storage as a complete response.

    Only captures fields the trajectory builder actually reads plus basic
    OpenAI envelope metadata (id/model/created). Trainer-facing fields only —
    we drop upstream-only fields like service_tier and system_fingerprint.
    """

    content: str = ""
    reasoning_content: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    response_logprobs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    response_id: str = ""
    model: Optional[str] = None
    created: Optional[int] = None
    input_token_ids: Optional[list[int]] = None
    prompt_token_ids: Optional[list[int]] = None

    def accumulate(self, chunk: dict[str, Any]) -> None:
        """Extract and accumulate data from a single streaming chunk."""
        self.chunks.append(chunk)
        self._capture_metadata(chunk)

        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        if self.input_token_ids is None:
            input_token_ids = choice.get("input_token_ids")
            if isinstance(input_token_ids, list):
                self.input_token_ids = input_token_ids

        if delta.get("content"):
            self.content += delta["content"]
        if delta.get("reasoning_content"):
            self.reasoning_content += delta["reasoning_content"]

        if delta.get("tool_calls"):
            self._merge_tool_calls(delta["tool_calls"])

        chunk_token_ids = choice.get("token_ids")
        if isinstance(chunk_token_ids, list):
            self.token_ids.extend(chunk_token_ids)

        logprobs_obj = choice.get("logprobs", {}) or {}
        if "content" in logprobs_obj:
            for token_info in logprobs_obj.get("content") or []:
                if isinstance(token_info, dict):
                    self.response_logprobs.append(token_info.copy())
                    if not chunk_token_ids and "token_id" in token_info:
                        self.token_ids.append(token_info["token_id"])

        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        if choice.get("stop_reason") is not None:
            self.stop_reason = choice["stop_reason"]

    def _capture_metadata(self, chunk: dict[str, Any]) -> None:
        if not self.response_id and chunk.get("id"):
            self.response_id = chunk["id"]
        if self.model is None and chunk.get("model"):
            self.model = chunk["model"]
        if self.created is None and chunk.get("created") is not None:
            self.created = chunk["created"]
        if self.prompt_token_ids is None and chunk.get("prompt_token_ids") is not None:
            self.prompt_token_ids = chunk["prompt_token_ids"]

    def _merge_tool_calls(self, tool_call_deltas: list[dict]) -> None:
        for delta in tool_call_deltas:
            idx = delta.get("index", 0)
            while len(self.tool_calls) <= idx:
                self.tool_calls.append(
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
            current = self.tool_calls[idx]
            if delta.get("id"):
                current["id"] = delta["id"]
            if "function" in delta:
                func = delta["function"]
                if func.get("name"):
                    current["function"]["name"] += func["name"]
                if func.get("arguments"):
                    current["function"]["arguments"] += func["arguments"]

    def to_response(self) -> dict[str, Any]:
        """Reconstruct a complete response dict from accumulated chunks."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": self.finish_reason,
        }
        if self.stop_reason is not None:
            choice["stop_reason"] = self.stop_reason
        if self.response_logprobs:
            choice["logprobs"] = {"content": self.response_logprobs}
        if self.token_ids:
            choice["token_ids"] = self.token_ids
        if self.input_token_ids is not None:
            choice["input_token_ids"] = self.input_token_ids

        response: dict[str, Any] = {
            "id": self.response_id or (self.chunks[0].get("id", "") if self.chunks else ""),
            "object": "chat.completion",
            "choices": [choice],
        }
        if self.created is not None:
            response["created"] = self.created
        if self.model is not None:
            response["model"] = self.model
        if self.prompt_token_ids is not None:
            response["prompt_token_ids"] = self.prompt_token_ids
        return response
