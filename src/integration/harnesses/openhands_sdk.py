"""OpenHands SDK harness — lightweight SDK-based agent."""

from __future__ import annotations

import json
import shlex

from integration.base import BaseHarness
from integration.models import AgentSpec
from runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class OpenHandsSdkHarness(BaseHarness):
    """Run OpenHands SDK agent via an embedded runner script."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._runner_script = "/tmp/arp_openhands_sdk_run.py"

    async def setup(self, runtime: BaseRuntime) -> None:
        # Write the embedded runner script
        script = _RUNNER_SCRIPT
        await runtime.exec(
            f"cat > {self._runner_script} << 'ARPSCRIPT'\n{script}\nARPSCRIPT\n"
            f"chmod +x {self._runner_script}"
        )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        model = self.model_name or "openai/gpt-4o"
        env: dict[str, str] = {
            **self.env,
            "LLM_MODEL": model,
            "AGENT_INSTRUCTION": instruction,
        }

        # Pass MCP servers as JSON env var
        if self.mcp_servers:
            servers = [
                {
                    "name": s.name,
                    "transport": s.transport,
                    **({"command": s.command} if s.command else {}),
                    **({"args": s.args} if s.args else {}),
                    **({"url": s.url} if s.url else {}),
                }
                for s in self.mcp_servers
            ]
            env["MCP_SERVERS_JSON"] = json.dumps(servers)

        # Pass skills path
        if self.skills_path:
            env["SKILL_PATHS"] = self.skills_path

        # Map settings to env
        for key, env_key in [
            ("max_iterations", "MAX_ITERATIONS"),
            ("temperature", "LLM_TEMPERATURE"),
            ("reasoning_effort", "REASONING_EFFORT"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                env[env_key] = str(value)

        return [
            ExecInput(
                command=(
                    'export LLM_API_KEY="$OPENAI_API_KEY" LLM_BASE_URL="$OPENAI_BASE_URL" && '
                    f"source /opt/openhands-sdk-venv/bin/activate 2>/dev/null; "
                    f"python {self._runner_script} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/openhands-sdk.txt"
                ),
                env=env,
            )
        ]


_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""Minimal OpenHands SDK runner for ARP."""
import os
import sys

def main():
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    try:
        from pydantic import SecretStr
        from openhands.sdk import Agent, Conversation
        from openhands.sdk.llm import LLM
    except ImportError as e:
        print(f"openhands-sdk not installed: {e}", file=sys.stderr)
        sys.exit(1)

    instruction = os.environ.get("AGENT_INSTRUCTION", "")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "30"))

    llm = LLM(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        drop_params=True,
    )

    agent = Agent(llm=llm)
    workspace = os.environ.get("WORKSPACE_BASE", "/arp/session/workspace")

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
    )
    conversation.send_message(instruction)
    conversation.run()
    conversation.close()

if __name__ == "__main__":
    main()
'''
