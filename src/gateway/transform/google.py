"""Google Generative AI API transformer.

Transforms between Google Generative AI API and OpenAI Chat Completions API.
Aligned with agent-harness-proxy/src/harness_proxy/transform/google.py.
"""

from __future__ import annotations

from typing import Any

from gateway.transform.base import BaseTransformer


class GoogleTransformer(BaseTransformer):
    """Transform between Google Generative AI and OpenAI API formats."""

    ROLE_MAP = {"user": "user", "model": "assistant"}
    FINISH_REASON_MAP_REVERSE = {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "content_filter": "SAFETY",
        "tool_calls": "STOP",
    }

    def transform_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = []

        # System instruction
        system_instruction = body.get("systemInstruction")
        if system_instruction:
            system_text = self._extract_text_from_parts(system_instruction.get("parts", []))
            if system_text:
                messages.append({"role": "system", "content": system_text})

        # Contents → messages
        for content in body.get("contents", []):
            role = content.get("role", "user")
            openai_role = self.ROLE_MAP.get(role, "user")
            text = self._extract_text_from_parts(content.get("parts", []))
            if text:
                messages.append({"role": openai_role, "content": text})

        result: dict[str, Any] = {"messages": messages}

        # Generation config
        gen_config = body.get("generationConfig", {})
        if "maxOutputTokens" in gen_config:
            result["max_tokens"] = gen_config["maxOutputTokens"]
        if "temperature" in gen_config:
            result["temperature"] = gen_config["temperature"]
        if "topP" in gen_config:
            result["top_p"] = gen_config["topP"]
        if "stopSequences" in gen_config:
            result["stop"] = gen_config["stopSequences"]

        # Streaming flag (Google uses different endpoint, detected from URL)
        if body.get("_streaming", False):
            result["stream"] = True

        return self._enhance_token_params(result)

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = []
        for i, choice in enumerate(response.get("choices", [])):
            message = choice.get("message", {})
            parts = []
            content = message.get("content")
            if content:
                parts.append({"text": content})

            finish_reason = choice.get("finish_reason", "stop")
            google_finish = self.FINISH_REASON_MAP_REVERSE.get(finish_reason, "STOP")

            candidates.append({
                "content": {"parts": parts, "role": "model"},
                "finishReason": google_finish,
                "index": i,
                "safetyRatings": [],
            })

        usage = response.get("usage", {})
        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            },
        }

    def transform_stream_chunk(
        self,
        chunk: dict[str, Any],
        original_request: dict[str, Any],
        is_first: bool = False,
    ) -> dict[str, Any]:
        candidates = []
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            parts = []
            content = delta.get("content")
            if content:
                parts.append({"text": content})

            candidate: dict[str, Any] = {
                "content": {"parts": parts, "role": "model"},
                "index": choice.get("index", 0),
            }

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                candidate["finishReason"] = self.FINISH_REASON_MAP_REVERSE.get(finish_reason, "STOP")

            candidates.append(candidate)

        result: dict[str, Any] = {"candidates": candidates}

        usage = chunk.get("usage")
        if usage:
            result["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }

        return result

    def is_streaming_request(self, body: dict[str, Any]) -> bool:
        return body.get("_streaming", False)

    def _extract_text_from_parts(self, parts: list) -> str:
        texts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
