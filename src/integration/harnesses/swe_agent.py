"""SWE-Agent harness — https://github.com/SWE-agent/SWE-agent"""

from __future__ import annotations

import shlex

from integration.base import BaseHarness
from integration.models import AgentRunResult, AgentSpec
from runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class SweAgentHarness(BaseHarness):
    """Run SWE-Agent CLI against a task."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._problem_statement_path = f"{RUNTIME_AGENT_LOG_DIR}/problem_statement.md"
        self._repo_path = str(self.settings.get("repo_path") or "/arp/session/workspace")
        self._shell_preamble = str(self.settings.get("shell_preamble") or "").strip()

    async def setup(self, runtime: BaseRuntime) -> None:
        pass

    def run_steps(self, instruction: str) -> list[ExecInput]:
        model = self.model_name or "openai/gpt-4o"
        env: dict[str, str] = {**self.env}

        flags: list[str] = []
        for key, cli in [
            ("per_instance_cost_limit", "--agent.model.per_instance_cost_limit"),
            ("total_cost_limit", "--agent.model.total_cost_limit"),
            ("max_input_tokens", "--agent.model.max_input_tokens"),
            ("temperature", "--agent.model.temperature"),
            ("top_p", "--agent.model.top_p"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")

        flags_str = (" " + " ".join(flags)) if flags else ""
        preamble = f"{self._shell_preamble} && " if self._shell_preamble else ""

        safe_instruction = instruction.replace("'", "'\"'\"'")

        return [
            ExecInput(
                command=(
                    f"cat > {self._problem_statement_path} << 'ARPINST'\n{safe_instruction}\nARPINST\n"
                    f"{preamble}"
                    'export OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" && '
                    f"sweagent run "
                    f"--agent.model.name={shlex.quote(model)} "
                    f"--problem_statement.path={shlex.quote(self._problem_statement_path)} "
                    f"--env.deployment.type=local "
                    f"--env.repo.path={shlex.quote(self._repo_path)}"
                    f"{flags_str} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/swe-agent.txt"
                ),
                env=env,
            )
        ]

    def cleanup_steps(self) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    f"cp /root/trajectories/**/*.traj "
                    f"{RUNTIME_AGENT_LOG_DIR}/swe-agent.trajectory.json 2>/dev/null || true"
                ),
            )
        ]
