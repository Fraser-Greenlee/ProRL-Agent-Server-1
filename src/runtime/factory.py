"""Runtime factory with built-in backend map and import_path support."""

from __future__ import annotations

import importlib
from pathlib import Path

from runtime.apptainer import ApptainerRuntime
from runtime.base import BaseRuntime
from runtime.docker import DockerRuntime
from runtime.models import RuntimeSpec

_BUILTIN_BACKENDS: dict[str, type[BaseRuntime]] = {
    "docker": DockerRuntime,
    "apptainer": ApptainerRuntime,
}


def create_runtime(
    spec: RuntimeSpec, session_id: str, session_dir: Path
) -> BaseRuntime:
    """Instantiate a runtime from a RuntimeSpec.

    Uses the built-in backend map for ``docker`` and ``apptainer``.
    Falls back to ``spec.import_path`` for plugin runtimes.
    """
    if spec.import_path:
        cls = _import_runtime_class(spec.import_path)
        runtime = cls(spec, session_id, session_dir)
        _validate_runtime_capabilities(runtime)
        return runtime
    cls = _BUILTIN_BACKENDS.get(spec.backend)
    if cls is None:
        raise ValueError(f"Unsupported runtime backend: {spec.backend}")
    runtime = cls(spec, session_id, session_dir)
    _validate_runtime_capabilities(runtime)
    return runtime


def _import_runtime_class(import_path: str) -> type[BaseRuntime]:
    module_name, sep, attr_name = import_path.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid runtime import path: {import_path!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseRuntime):
        raise TypeError(f"{import_path} is not a subclass of BaseRuntime")
    return cls


def _validate_runtime_capabilities(runtime: BaseRuntime) -> None:
    spec = runtime.spec
    backend = spec.backend
    if spec.gpus > 0 and not runtime.supports_gpus:
        raise ValueError(f"runtime backend {backend!r} does not support GPUs")
    if spec.cpus is not None and not runtime.supports_cpu_limits:
        raise ValueError(f"runtime backend {backend!r} does not support CPU limits")
    if spec.memory_mb is not None and not runtime.supports_memory_limits:
        raise ValueError(f"runtime backend {backend!r} does not support memory limits")
    if spec.storage_mb is not None and not runtime.supports_storage_limits:
        raise ValueError(f"runtime backend {backend!r} does not support storage limits")
    if not spec.allow_internet:
        if not runtime.can_disable_internet:
            raise ValueError(
                f"runtime backend {backend!r} cannot disable internet access"
            )
        if spec.network not in (None, "", "host", "none"):
            raise ValueError(
                "runtime.network must be unset, 'host', or 'none' when "
                "allow_internet=false"
            )
