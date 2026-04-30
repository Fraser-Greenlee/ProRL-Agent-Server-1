"""Qwen Code harness — https://github.com/QwenLM/qwen-code"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class QwenCodeHarness(BaseHarness):
    """Run Qwen Code CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._qwen_dir = "$HOME/.qwen"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._qwen_dir}")

        # Register MCP servers
        if self.mcp_servers:
            servers_config: list[dict] = []
            for server in self.mcp_servers:
                entry: dict = {"name": server.name, "transport": server.transport}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry["url"] = server.url
                servers_config.append(entry)
            config = {"mcpServers": servers_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._qwen_dir}/settings.json << 'POLARCFG'\n{config_json}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._qwen_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._qwen_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {**self.env}
        # qwen-code can load persisted OpenAI credentials from the user config.
        # Pin the proxy credentials on the command line so every completion is
        # recorded under the Polar session rather than a host API key.
        model_flag = ""
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name
            model_flag = f"--model {shlex.quote(self.model_name)} "

        return [
            ExecInput(
                command=(
                    'OPENAI_API_KEY="$SESSION_ID" '
                    'OPENAI_BASE_URL="$OPENAI_BASE_URL" '
                    "QWEN_DEFAULT_AUTH_TYPE=openai "
                    "qwen --auth-type openai "
                    '--openai-api-key "$SESSION_ID" '
                    '--openai-base-url "$OPENAI_BASE_URL" '
                    '--max-session-turns "${QWEN_CODE_MAX_SESSION_TURNS:-40}" '
                    f"{model_flag}--yolo --prompt={escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/qwen-code.txt"
                ),
                env=env,
            )
        ]
