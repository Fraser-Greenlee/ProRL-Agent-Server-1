"""Cursor CLI harness — https://cursor.com"""

from __future__ import annotations

import json
import shlex

from integration.base import BaseHarness
from integration.models import AgentSpec
from runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class CursorCliHarness(BaseHarness):
    """Run Cursor CLI agent in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._cursor_dir = "/root/.cursor"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._cursor_dir}")

        # Register MCP servers
        if self.mcp_servers:
            mcp_config: dict[str, dict] = {}
            for server in self.mcp_servers:
                entry: dict = {}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry["url"] = server.url
                    entry["type"] = server.transport
                mcp_config[server.name] = entry
            config = {"mcpServers": mcp_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._cursor_dir}/mcp.json << 'ARPCFG'\n{config_json}\nARPCFG"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {**self.env}

        flags: list[str] = [
            "--yolo",
            "--print",
            "--output-format=stream-json",
        ]
        if self.model_name:
            flags.append(f"--model={shlex.quote(self.model_name)}")

        mode = self.settings.get("mode")
        if mode:
            flags.append(f"--{mode}")

        flags_str = " ".join(flags)
        return [
            ExecInput(
                command=(
                    'export PATH="$HOME/.local/bin:$PATH" '
                    'CURSOR_API_KEY="$OPENAI_API_KEY" && '
                    f"cursor-agent {flags_str} -- {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/cursor-cli.txt"
                ),
                env=env,
            )
        ]
