import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime


@dataclass
class JobDetails:
    job_id: str | None = None
    instance: pd.Series | None = None
    max_iterations: int = 2
    llm_config: LLMConfig | None = None
    runtime: Runtime | None = None
    metadata: EvalMetadata | None = None
    config: OpenHandsConfig | None = None
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event | None = None
    start_time: float | None = None
    start_run_time: float | None = None
    start_eval_time: float | None = None
    end_time: float | None = None


class AgentHandler(ABC):
    """Abstract base class defining the interface for agent handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name identifier for this agent handler."""
        pass

    @abstractmethod
    async def init(
        self,
        instance: pd.Series,
        llm_config: LLMConfig | None = None,
        sid: str | None = None,
        max_iterations: int = 1,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the agent with instance and config."""
        pass

    @abstractmethod
    async def run(
        self,
        runtime: Runtime,
        metadata: EvalMetadata,
        config: OpenHandsConfig,
        instance: pd.Series,
    ) -> dict[str, object]:
        """Run the agent with runtime and instance."""
        pass

    @abstractmethod
    async def eval(
        self, job_details: JobDetails, sid: str | None = None, allow_skip: bool = True
    ) -> dict[str, Any]:
        """Evaluate the agent results."""
        pass

    @abstractmethod
    def init_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during initialization."""
        pass

    @abstractmethod
    def run_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during run."""
        pass

    @abstractmethod
    def eval_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during evaluation."""
        pass

    @abstractmethod
    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results."""
        pass


# Registry for different types of agent functions
_registries: dict[str, dict[str, Callable[..., Any]]] = {
    'init': {},
    'run': {},
    'eval': {},
    'init_exception': {},
    'run_exception': {},
    'eval_exception': {},
    'final_result': {},
}

_registered_handlers: set[str] = set()


class FunctionNotRegisteredError(Exception):
    """Raised when a requested function is not found in the registry."""

    pass


# Utility functions for registry management
def get_registered_functions(registry_type, name):
    return _registries.get(registry_type, {}).get(name)


def is_registered_handler(name):
    """Check if a handler is registered correctly."""
    return name in _registered_handlers


def register_agent_handler(handler: AgentHandler):
    """Register all methods of an AgentHandler instance to their corresponding registries."""
    _registered_handlers.add(handler.name)
    _registries['init'][handler.name] = handler.init
    _registries['run'][handler.name] = handler.run
    _registries['eval'][handler.name] = handler.eval
    _registries['init_exception'][handler.name] = handler.init_exception
    _registries['run_exception'][handler.name] = handler.run_exception
    _registries['eval_exception'][handler.name] = handler.eval_exception
    _registries['final_result'][handler.name] = handler.final_result
