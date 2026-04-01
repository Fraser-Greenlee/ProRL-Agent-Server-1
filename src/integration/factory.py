"""Harness factory with built-in name map and import_path support."""

from __future__ import annotations

import importlib

from integration.base import BaseHarness
from integration.models import AgentSpec


def _builtin_harness_map() -> dict[str, type[BaseHarness]]:
    """Lazy import to avoid circular imports at module level."""
    from integration.harnesses.claude_code import ClaudeCodeHarness
    from integration.harnesses.codex import CodexHarness
    from integration.harnesses.gemini_cli import GeminiCliHarness
    from integration.harnesses.openhands_sdk import OpenHandsSdkHarness
    from integration.harnesses.opencode import OpenCodeHarness
    from integration.harnesses.qwen_code import QwenCodeHarness
    from integration.harnesses.shell import ShellHarness
    from integration.harnesses.swe_agent import SweAgentHarness

    return {
        "claude_code": ClaudeCodeHarness,
        "codex": CodexHarness,
        "gemini_cli": GeminiCliHarness,
        "openhands_sdk": OpenHandsSdkHarness,
        "opencode": OpenCodeHarness,
        "qwen_code": QwenCodeHarness,
        "shell": ShellHarness,
        "swe_agent": SweAgentHarness,
    }


def create_harness(agent_spec: AgentSpec) -> BaseHarness:
    """Resolve and instantiate a harness from an AgentSpec."""
    if agent_spec.import_path is not None:
        cls = _import_harness_class(agent_spec.import_path)
        return cls(agent_spec)

    if agent_spec.harness is not None:
        harness_map = _builtin_harness_map()
        cls = harness_map.get(agent_spec.harness)
        if cls is None:
            raise ValueError(f"Unknown harness: {agent_spec.harness!r}")
        return cls(agent_spec)

    raise ValueError("AgentSpec must specify harness or import_path")


def _import_harness_class(import_path: str) -> type[BaseHarness]:
    module_name, sep, attr_name = import_path.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid harness import path: {import_path!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseHarness):
        raise TypeError(f"{import_path} is not a subclass of BaseHarness")
    return cls
