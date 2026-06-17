"""Unified Hermes harness — Hermes with a fixed toolset + browser/search config.

A thin variant of :class:`HermesHarness` that bakes in the toolset preset used by
the R2E-Gym / LiteResearcher / CUA-Gym rollout examples: the full
``terminal,file,code_execution,browser,web,search`` toolset, a browser block
wired for the in-runtime apps (private URLs allowed, Chromium engine), a 256k
context window, and the stateless ``HERMES_SKIP_*`` env. Tasks still override any
of these through ``settings``/``env`` and inject their own ``mcp_servers``.
"""

from __future__ import annotations

from polar.agent.presets.hermes import HermesHarness
from polar.agent.models import AgentSpec
from polar.runtime.models import ExecInput


class NotreHermesHarness(HermesHarness):
    """Hermes preset with the predefined unified toolset, browser, and search."""

    DEFAULT_TOOLSETS = "terminal,file,code_execution,browser,web,search"
    DEFAULT_CONTEXT_LENGTH = 262_144
    # Hermes runs from an isolated venv (see the examples' install step); keep its
    # imports off the runtime's user site-packages, and run stateless.
    DEFAULT_ENV = {
        "HERMES_SKIP_CONTEXT_FILES": "1",
        "HERMES_SKIP_MEMORY": "1",
        "HERMES_ALLOW_PRIVATE_URLS": "true",
        "AGENT_BROWSER_ENGINE": "chrome",
        "PYTHONNOUSERSITE": "1",
    }

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Task settings/env win over the preset defaults.
        self.settings = {
            "toolsets": self.DEFAULT_TOOLSETS,
            "context_length": self.DEFAULT_CONTEXT_LENGTH,
            **self.settings,
        }
        self.env = {**self.DEFAULT_ENV, **self.env}

    def _build_config(self) -> dict:
        config = super()._build_config()
        config["model"] = {
            "context_length": int(
                self.settings.get("context_length", self.DEFAULT_CONTEXT_LENGTH)
            )
        }
        config["browser"] = {
            "allow_private_urls": True,
            "auto_local_for_private_urls": True,
            "command_timeout": int(self.settings.get("browser_command_timeout", 90)),
            "engine": "chrome",
        }
        config["security"] = {"allow_private_urls": True}
        return config

    def postrun_steps(self) -> list[ExecInput]:
        # CUA-Gym uses this to kill the per-task Vite dev server after eval.
        commands = self.settings.get("postrun_commands") or []
        if not isinstance(commands, list):
            return []
        return [ExecInput(command=str(c)) for c in commands if str(c).strip()]
