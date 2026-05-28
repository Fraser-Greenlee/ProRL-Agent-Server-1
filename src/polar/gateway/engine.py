"""Inference backend strategies for the Polar gateway.

The gateway speaks the OpenAI Chat Completions API to a local inference server.
Two backends are supported, and they differ only in:

  1. the request params that make them emit the token ids + per-token logprobs
     Polar needs for training, and
  2. the exact shape of those fields in the response.

Each backend is a small strategy with two hooks: ``prepare_request`` (inject
engine-specific params) and ``normalize_response`` (canonicalize the response
so everything downstream -- storage, trace builder, transforms, slime adapter
-- sees one shape). The canonical shape is SGLang's patched output:

  - prompt token ids:   ``choice.input_token_ids`` (or ``response.prompt_token_ids``)
  - response token ids: ``choice.token_ids``       (or ``logprobs.content[].token_id``)
  - per-token logprobs: ``choice.logprobs.content[]`` with ``{token, token_id, logprob, ...}``

SGLang reaches this shape via ``scripts/patch/patch_sglang.sh``; vLLM reaches it
natively via the ``return_token_ids`` request flag plus a light response rename.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceEngine(ABC):
    """Strategy for one OpenAI-compatible inference backend."""

    name: str

    @abstractmethod
    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject the params this backend needs to emit training token ids/logprobs."""

    @abstractmethod
    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize the backend's response (in place) and return it."""


class SGLangEngine(InferenceEngine):
    """SGLang is already canonical (via patch_sglang.sh); both hooks pass through."""

    name = "sglang"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        return response


class VLLMEngine(InferenceEngine):
    """vLLM via its native OpenAI-compatible server.

    ``return_token_ids`` makes vLLM emit ``response.prompt_token_ids`` and
    ``choice.token_ids`` -- the same ids SGLang's patch produces, with no source
    patch needed. ``top_logprobs`` must be set (not None) for vLLM to populate
    ``logprobs.content[]`` given ``logprobs=True``; 0 returns just the sampled
    token's logprob, which is all training needs.
    """

    name = "vllm"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request["return_token_ids"] = True
        if request.get("logprobs"):
            request.setdefault("top_logprobs", 0)
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_reasoning(choice.get("message"))
            self._stamp_token_ids_onto_logprobs(choice)
        return response

    @staticmethod
    def _canonicalize_reasoning(message: Any) -> None:
        """vLLM names the field ``reasoning``; Polar's canonical field is ``reasoning_content``."""
        if not isinstance(message, dict):
            return
        if message.get("reasoning_content") is None and message.get("reasoning") is not None:
            message["reasoning_content"] = message.pop("reasoning")

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Parity with SGLang: copy token_id onto each logprob entry.

        vLLM builds ``logprobs.content`` and ``choice.token_ids`` from the same
        ``output.token_ids``, so they align; guard on equal length regardless.
        Not load-bearing for training (which reads ``choice.token_ids`` and the
        per-entry ``logprob``) -- it keeps stored traces one shape across engines.
        """
        token_ids = choice.get("token_ids")
        logprobs = choice.get("logprobs")
        if not isinstance(token_ids, list) or not isinstance(logprobs, dict):
            return
        content = logprobs.get("content")
        if not isinstance(content, list) or len(content) != len(token_ids):
            return
        for entry, token_id in zip(content, token_ids):
            if isinstance(entry, dict):
                entry.setdefault("token_id", token_id)


_ENGINES: dict[str, type[InferenceEngine]] = {
    SGLangEngine.name: SGLangEngine,
    VLLMEngine.name: VLLMEngine,
}


def get_engine(name: str) -> InferenceEngine:
    """Return the inference engine strategy for ``name`` (``sglang`` | ``vllm``)."""
    try:
        engine_cls = _ENGINES[name]
    except KeyError:
        supported = ", ".join(sorted(_ENGINES))
        raise ValueError(
            f"Unknown inference engine {name!r}; supported: {supported}"
        ) from None
    return engine_cls()
