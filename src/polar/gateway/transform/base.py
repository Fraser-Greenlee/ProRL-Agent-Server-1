"""Base transformer interface with vLLM token collection enhancement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTransformer(ABC):
    """Abstract base class for API transformers.

    Transforms requests from source API format to OpenAI format (for vLLM),
    and transforms responses back to source API format.
    """

    @abstractmethod
    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Transform request body to OpenAI/vLLM format."""
        pass

    @abstractmethod
    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Transform response back to source API format."""
        pass

    @abstractmethod
    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Transform a streaming chunk to source API format."""
        pass

    def is_streaming_request(self, body: dict[str, Any]) -> bool:
        """Check if request is for streaming response."""
        return body.get("stream", False)

    def create_stream_state(self, original_request: dict[str, Any]) -> Any | None:
        """Create per-request stream state when chunk transforms need memory."""
        return None

    def _enhance_token_params(self, request: dict[str, Any]) -> dict[str, Any]:
        """Hardcode vLLM token collection params for RL training data.

        - logprobs: enable log probability collection
        - return_token_ids: return raw token ID sequences
        - include_stop_str_in_output: include stop strings in output
        - stream_options.include_usage: return usage in streaming
        """
        request["logprobs"] = True
        request["include_stop_str_in_output"] = True
        request["return_token_ids"] = True
        if request.get("stream"):
            request.setdefault("stream_options", {})["include_usage"] = True
        return request
