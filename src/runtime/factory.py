"""Runtime factory with built-in backend map and import_path support."""

from __future__ import annotations

import importlib
from pathlib import Path

from runtime.base import BaseRuntime
from runtime.docker import DockerRuntime
from runtime.models import RuntimeSpec
from runtime.singularity import SingularityRuntime

_BUILTIN_BACKENDS: dict[str, type[BaseRuntime]] = {
    "docker": DockerRuntime,
    "singularity": SingularityRuntime,
}


def create_runtime(
    spec: RuntimeSpec, session_id: str, session_dir: Path
) -> BaseRuntime:
    """Instantiate a runtime from a RuntimeSpec.

    Uses the built-in backend map for ``docker`` and ``singularity``.
    Falls back to ``spec.import_path`` for plugin runtimes.
    """
    if spec.import_path:
        cls = _import_runtime_class(spec.import_path)
        return cls(spec, session_id, session_dir)
    cls = _BUILTIN_BACKENDS.get(spec.backend)
    if cls is None:
        raise ValueError(f"Unsupported runtime backend: {spec.backend}")
    return cls(spec, session_id, session_dir)


def _import_runtime_class(import_path: str) -> type[BaseRuntime]:
    module_name, sep, attr_name = import_path.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid runtime import path: {import_path!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseRuntime):
        raise TypeError(f"{import_path} is not a subclass of BaseRuntime")
    return cls
