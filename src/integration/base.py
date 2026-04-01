"""Base harness contract for agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from integration.models import AgentRunResult, AgentSpec, MCPServerSpec
from runtime.base import BaseRuntime
from runtime.models import ExecInput


class BaseHarness(ABC):
    """Abstract base for all agent harnesses.

    Each harness converts structured AgentSpec configuration into a sequence
    of ExecInput commands that the gateway node runs inside a runtime.
    """

    def __init__(
        self,
        agent_spec: AgentSpec,
    ) -> None:
        self.agent_spec = agent_spec
        self.model_name = agent_spec.model_name
        self.settings = agent_spec.settings
        self.env = agent_spec.env
        self.mcp_servers = agent_spec.mcp_servers
        self.skills_path = agent_spec.skills_path

    async def setup(self, runtime: BaseRuntime) -> None:
        """Optional setup step run before the agent.

        Override to install packages, write config files, etc.
        """

    @abstractmethod
    def run_steps(self, instruction: str) -> list[ExecInput]:
        """Return the ordered list of commands to execute the agent task."""

    def cleanup_steps(self) -> list[ExecInput]:
        """Return commands to run best-effort in finally after the agent."""
        return []

    def teardown_steps(self) -> list[ExecInput]:
        """Return best-effort commands to run after evaluation has finished."""
        return self.cleanup_steps()

    async def postprocess(
        self, runtime: BaseRuntime, result: AgentRunResult
    ) -> None:
        """Optional post-processing after agent execution completes."""
