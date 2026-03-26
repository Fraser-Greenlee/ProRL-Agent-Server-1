"""Shell harness — first-class escape hatch for agents defined as shell commands."""

from __future__ import annotations

from integration.base import BaseHarness
from integration.models import AgentSpec
from runtime.models import ExecInput


class ShellHarness(BaseHarness):
    """Execute a single ExecInput from AgentSpec.custom_shell."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        if agent_spec.custom_shell is None:
            raise ValueError("ShellHarness requires custom_shell in AgentSpec")
        self._shell = agent_spec.custom_shell

    def run_steps(self, instruction: str) -> list[ExecInput]:
        return [self._shell]
