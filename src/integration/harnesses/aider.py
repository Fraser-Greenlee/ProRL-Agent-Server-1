"""Aider harness — https://aider.chat/"""

from __future__ import annotations

import shlex

from integration.base import BaseHarness
from runtime.base import RUNTIME_AGENT_LOG_DIR
from runtime.models import ExecInput


class AiderHarness(BaseHarness):
    """Run Aider in scripted --yes mode."""

    def run_steps(self, instruction: str) -> list[ExecInput]:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model_name must be in provider/model format for Aider")

        provider = self.model_name.split("/", 1)[0]
        # Pass full model name to aider so litellm can route correctly
        model = self.model_name
        escaped = shlex.quote(instruction)

        flags: list[str] = []
        for key, cli in [
            ("reasoning_effort", "--reasoning-effort"),
            ("thinking_tokens", "--thinking-tokens"),
            ("test_cmd", "--test-cmd"),
            ("map_tokens", "--map-tokens"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")
        for key, cli in [
            ("cache_prompts", "--cache-prompts"),
            ("auto_lint", "--auto-lint"),
            ("auto_test", "--auto-test"),
            ("stream", "--stream"),
        ]:
            value = self.settings.get(key)
            if value is True:
                flags.append(cli)
            elif value is False:
                flags.append(f"--no-{cli.lstrip('-')}")

        flags_str = (" ".join(flags) + " ") if flags else ""

        # Set AIDER_API_KEY via shell expansion in the command
        if provider == "anthropic":
            key_export = 'export AIDER_API_KEY="anthropic=$ANTHROPIC_API_KEY"'
        else:
            key_export = f'export AIDER_API_KEY="{provider}=$OPENAI_API_KEY"'

        return [
            ExecInput(
                command=(
                    f"{key_export} && "
                    f"aider --yes {flags_str}--model={shlex.quote(model)} "
                    f"--message={escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/aider.txt"
                ),
                env=dict(self.env),
            )
        ]
