"""OpenHands harness — https://github.com/All-Hands-AI/OpenHands"""

from __future__ import annotations

import shlex

from integration.base import BaseHarness
from integration.models import AgentSpec
from runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class OpenHandsHarness(BaseHarness):
    """Run OpenHands core agent (non-SDK) in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)

    async def setup(self, runtime: BaseRuntime) -> None:
        # Create skills dir that OpenHands expects
        site_pkg = await runtime.exec(
            'python -c "import site; print(site.getsitepackages()[0])"'
        )
        if site_pkg.return_code == 0 and site_pkg.stdout:
            pkg_dir = site_pkg.stdout.strip()
            await runtime.exec(f"mkdir -p {pkg_dir}/skills")

        # Write MCP config if servers provided
        if self.mcp_servers:
            toml_lines: list[str] = []
            for server in self.mcp_servers:
                toml_lines.append(f'[[mcp_servers."{server.name}"]]')
                toml_lines.append(f'transport = "{server.transport}"')
                if server.transport == "stdio":
                    toml_lines.append(f'command = "{server.command}"')
                    if server.args:
                        args_str = ", ".join(f'"{a}"' for a in server.args)
                        toml_lines.append(f"args = [{args_str}]")
                else:
                    toml_lines.append(f'url = "{server.url}"')
            toml_content = "\n".join(toml_lines)
            await runtime.exec(
                f"mkdir -p /root/.openhands && "
                f"cat > /root/.openhands/mcp_config.toml << 'ARPCFG'\n{toml_content}\nARPCFG"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)

        model = self.model_name or "openai/gpt-4o"
        env: dict[str, str] = {
            **self.env,
            "LLM_MODEL": model,
            "SANDBOX_VOLUMES": "",
        }

        # Map settings to env vars
        for key, env_key in [
            ("max_iterations", "MAX_ITERATIONS"),
            ("temperature", "LLM_TEMPERATURE"),
            ("disable_tool_calls", "DISABLE_TOOL_CALLS"),
            ("reasoning_effort", "REASONING_EFFORT"),
            ("caching_prompt", "LLM_CACHING_PROMPT"),
            ("top_p", "LLM_TOP_P"),
            ("num_retries", "LLM_NUM_RETRIES"),
            ("max_budget_per_task", "MAX_BUDGET_PER_TASK"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                env[env_key] = str(value)

        return [
            ExecInput(
                command=(
                    'export LLM_BASE_URL="$OPENAI_BASE_URL" LLM_API_KEY="$OPENAI_API_KEY" '
                    'SANDBOX_TYPE=local WORKSPACE_BASE=/arp/session/workspace && '
                    f"python -m openhands.core.main "
                    f"--task={escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/openhands.txt"
                ),
                env=env,
            )
        ]
