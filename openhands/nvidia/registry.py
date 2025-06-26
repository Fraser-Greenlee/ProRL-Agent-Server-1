import threading
from dataclasses import dataclass

import pandas as pd

from evaluation.utils.shared import EvalMetadata
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime


@dataclass
class JobDetails:
    job_id: str = None
    instance: pd.Series = None
    max_iterations: int = 2
    llm_config: LLMConfig = None
    runtime: Runtime = None
    metadata: EvalMetadata = None
    config: OpenHandsConfig = None
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event = None
    start_time: float = None
    start_run_time: float = None
    start_eval_time: float = None
    end_time: float = None


# Registry for different types of agent functions
_registries = {
    'init': {},
    'run': {},
    'eval': {},
    'init_exception': {},
    'run_exception': {},
    'eval_exception': {},
    'final_result': {},
}


def _create_register_decorator(registry_type):
    """Factory function to create registration decorators."""

    def register_func(name):
        def decorator(func):
            _registries[registry_type][name] = func
            return func

        return decorator

    return register_func


# Create registration decorators
register_init_func = _create_register_decorator('init')
register_run_func = _create_register_decorator('run')
register_eval_func = _create_register_decorator('eval')
register_init_exception_func = _create_register_decorator('init_exception')
register_run_exception_func = _create_register_decorator('run_exception')
register_eval_exception_func = _create_register_decorator('eval_exception')
register_final_result_func = _create_register_decorator('final_result')


class FunctionNotRegisteredError(Exception):
    """Raised when a requested function is not found in the registry."""

    pass


# Utility functions for registry management
def get_registered_functions(registry_type, name):
    return _registries.get(registry_type, {}).get(name)


def clear_registry(registry_type=None):
    """Clear one or all registries."""
    if registry_type:
        _registries[registry_type].clear()
    else:
        for registry in _registries.values():
            registry.clear()


def is_registered(name, registry_type):
    """Check if a function is registered in a specific registry."""
    return name in _registries.get(registry_type, {})
