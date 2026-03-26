"""OpenCode harness — https://github.com/opencode-ai/opencode"""

from __future__ import annotations

import json
import shlex

from integration.base import BaseHarness
from integration.models import AgentSpec
from runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class OpenCodeHarness(BaseHarness):
    """Run OpenCode CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._config_dir = "/root/.config/opencode"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._config_dir}")

        if not self.model_name:
            return

        # Build opencode.json config with provider/model
        provider, model_id = (
            self.model_name.split("/", 1)
            if "/" in self.model_name
            else ("openai", self.model_name)
        )

        config: dict = {
            "provider": {provider: {"models": {model_id: {}}}},
        }

        # Register MCP servers
        if self.mcp_servers:
            mcp_config: dict = {}
            for server in self.mcp_servers:
                entry: dict = {"type": server.transport}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry["url"] = server.url
                mcp_config[server.name] = entry
            config["mcp"] = mcp_config

        config_json = json.dumps(config, indent=2)
        await runtime.exec(
            f"cat > {self._config_dir}/opencode.json << 'ARPCFG'\n{config_json}\nARPCFG"
        )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._config_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._config_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        model = self.model_name or "openai/gpt-4o"
        env: dict[str, str] = {
            **self.env,
            "OPENCODE_FAKE_VCS": "git",
        }

        return [
            ExecInput(
                command=(
                    f"opencode -m {shlex.quote(model)} run "
                    f"--format=json -- {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/opencode.txt"
                ),
                env=env,
            )
        ]
