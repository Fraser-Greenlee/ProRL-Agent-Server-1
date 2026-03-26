"""Configuration loading from YAML + environment variable overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from runtime.models import RuntimeSpec


@dataclass(frozen=True, slots=True)
class GatewayNodeConfig:
    id: str
    host: str
    port: int
    public_url: str
    model_served: str
    vllm_base_url: str
    vllm_timeout: float
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    ready_buffer_target: int
    default_runtime: RuntimeSpec | None = None


class Config:
    def __init__(self, config_path: str | None = None, *, require_existing: bool = False):
        self._data = self._load_data(config_path, require_existing=require_existing)
        self._validate()

    @classmethod
    def from_environment(cls) -> "Config":
        config_path = os.environ.get("CONFIG_PATH")
        if config_path is None:
            return cls("config.yaml")
        return cls(config_path, require_existing=True)

    @staticmethod
    def _load_data(config_path: str | None, *, require_existing: bool) -> dict[str, Any]:
        if not config_path:
            if require_existing:
                raise ValueError("Gateway config path cannot be empty")
            return {}

        path = Path(config_path)
        if not path.exists():
            if require_existing:
                raise FileNotFoundError(f"Gateway config file not found: {config_path}")
            return {}
        if not path.is_file():
            raise ValueError(f"Gateway config path is not a file: {config_path}")

        try:
            with path.open() as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid gateway YAML in {config_path}: {exc}") from exc

        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Gateway config {config_path} must contain a top-level mapping"
            )
        return loaded

    @staticmethod
    def _require_non_empty_string(value: object, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required")
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field_name} must be a non-empty string")
        return text

    @staticmethod
    def _coerce_port(value: object, field_name: str) -> int:
        try:
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"{field_name} must be between 1 and 65535")
        return port

    @staticmethod
    def _coerce_positive_int(value: object, field_name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0")
        return parsed

    @staticmethod
    def _coerce_positive_float(value: object, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number") from exc
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0")
        return parsed

    @classmethod
    def _coerce_http_url(cls, value: object, field_name: str) -> str:
        text = cls._require_non_empty_string(value, field_name)
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{field_name} must be an http:// or https:// URL")
        return text.rstrip("/")

    def _validate(self) -> None:
        self.gateway_nodes
        self.rollout_server_url
        self.heartbeat_interval_seconds

    @staticmethod
    def _default_public_url(host: str, port: int) -> str:
        public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{public_host}:{port}"

    def _legacy_gateway_node_entry(self) -> dict[str, Any]:
        legacy_capacity = self._data.get("node", {}).get("capacity", 1)
        return {
            "id": self._data.get("node", {}).get("id", socket.gethostname()),
            "host": self._data.get("server", {}).get("host", "0.0.0.0"),
            "port": self._data.get("server", {}).get("port", 8080),
            "public_url": self._data.get("server", {}).get("public_url"),
            "model_served": self._data.get("model_served", ""),
            "max_init_workers": self._data.get("node", {}).get("max_init_workers", 4),
            "max_run_workers": self._data.get("node", {}).get("max_run_workers", legacy_capacity),
            "max_postrun_workers": self._data.get("node", {}).get("max_postrun_workers", 4),
            "ready_buffer_target": self._data.get("node", {}).get("ready_buffer_target", legacy_capacity),
            "default_runtime": self._data.get("node", {}).get("default_runtime"),
            "vllm_backend": self._data.get("vllm", {}),
        }

    @cached_property
    def gateway_nodes(self) -> list[GatewayNodeConfig]:
        configured = self._data.get("gateway_nodes")
        if configured is None:
            configured = [self._legacy_gateway_node_entry()]
        if not isinstance(configured, list):
            raise ValueError("gateway_nodes must be a list")
        if not configured:
            raise ValueError("gateway_nodes must include at least one node")

        nodes: list[GatewayNodeConfig] = []
        seen_node_ids: set[str] = set()
        for index, entry in enumerate(configured):
            if not isinstance(entry, dict):
                raise ValueError(f"gateway_nodes[{index}] must be a mapping")

            raw_node_id = entry["id"] if "id" in entry else socket.gethostname()
            node_id = self._require_non_empty_string(
                raw_node_id,
                f"gateway_nodes[{index}].id",
            )
            if node_id in seen_node_ids:
                raise ValueError(f"Duplicate gateway node id: {node_id}")
            seen_node_ids.add(node_id)

            host = self._require_non_empty_string(
                entry["host"] if "host" in entry else "0.0.0.0",
                f"gateway_nodes[{index}].host",
            )
            port = self._coerce_port(
                entry["port"] if "port" in entry else 8080,
                f"gateway_nodes[{index}].port",
            )
            configured_public_url = entry.get("public_url")
            public_url = (
                self._coerce_http_url(
                    configured_public_url,
                    f"gateway_nodes[{index}].public_url",
                )
                if configured_public_url is not None
                else self._default_public_url(host, port)
            )
            max_run_workers = self._coerce_positive_int(
                entry["max_run_workers"] if "max_run_workers" in entry else entry.get("capacity", 1),
                f"gateway_nodes[{index}].max_run_workers",
            )
            max_init_workers = self._coerce_positive_int(
                entry["max_init_workers"] if "max_init_workers" in entry else 4,
                f"gateway_nodes[{index}].max_init_workers",
            )
            max_postrun_workers = self._coerce_positive_int(
                entry["max_postrun_workers"] if "max_postrun_workers" in entry else 4,
                f"gateway_nodes[{index}].max_postrun_workers",
            )
            ready_buffer_target = self._coerce_positive_int(
                entry["ready_buffer_target"] if "ready_buffer_target" in entry else max_run_workers,
                f"gateway_nodes[{index}].ready_buffer_target",
            )

            backend = entry["vllm_backend"] if "vllm_backend" in entry else {}
            if not isinstance(backend, dict):
                raise ValueError(f"gateway_nodes[{index}].vllm_backend must be a mapping")

            vllm_base_url = self._coerce_http_url(
                backend["base_url"] if "base_url" in backend else "http://localhost:8000",
                f"gateway_nodes[{index}].vllm_backend.base_url",
            )
            vllm_timeout = self._coerce_positive_float(
                backend["timeout"] if "timeout" in backend else 300,
                f"gateway_nodes[{index}].vllm_backend.timeout",
            )
            default_runtime_raw = entry.get("default_runtime")
            default_runtime = None
            if default_runtime_raw is not None:
                if not isinstance(default_runtime_raw, dict):
                    raise ValueError(f"gateway_nodes[{index}].default_runtime must be a mapping")
                default_runtime = RuntimeSpec.model_validate(default_runtime_raw)

            nodes.append(
                GatewayNodeConfig(
                    id=node_id,
                    host=host,
                    port=port,
                    public_url=public_url,
                    model_served=str(
                        entry.get("model_served", self._data.get("model_served", ""))
                    ),
                    vllm_base_url=vllm_base_url,
                    vllm_timeout=vllm_timeout,
                    max_init_workers=max_init_workers,
                    max_run_workers=max_run_workers,
                    max_postrun_workers=max_postrun_workers,
                    ready_buffer_target=ready_buffer_target,
                    default_runtime=default_runtime,
                )
            )
        return nodes

    @property
    def has_multiple_gateway_nodes(self) -> bool:
        return len(self.gateway_nodes) > 1

    @property
    def selected_gateway_node(self) -> GatewayNodeConfig:
        selector = os.environ.get("GATEWAY_NODE_ID")
        if selector:
            match = next((node for node in self.gateway_nodes if node.id == selector), None)
            if match is None:
                if len(self.gateway_nodes) == 1:
                    match = replace(self.gateway_nodes[0], id=selector)
                else:
                    raise ValueError(f"Unknown gateway node id {selector!r}")
        else:
            if len(self.gateway_nodes) != 1:
                raise ValueError(
                    "config contains multiple gateway_nodes; set GATEWAY_NODE_ID "
                    "or run `python -m gateway.server` to launch all configured nodes"
                )
            match = self.gateway_nodes[0]

        host = self._require_non_empty_string(
            os.environ.get("PROXY_HOST", match.host),
            "PROXY_HOST or gateway_nodes[].host",
        )
        port = self._coerce_port(
            os.environ.get("PROXY_PORT", match.port),
            "PROXY_PORT or gateway_nodes[].port",
        )
        configured_public_url = os.environ.get("PROXY_PUBLIC_URL")
        if configured_public_url:
            public_url = self._coerce_http_url(
                configured_public_url,
                "PROXY_PUBLIC_URL",
            )
        elif "PROXY_HOST" in os.environ or "PROXY_PORT" in os.environ:
            public_url = self._default_public_url(host, port)
        else:
            public_url = match.public_url

        return GatewayNodeConfig(
            id=match.id,
            host=host,
            port=port,
            public_url=public_url,
            model_served=str(os.environ.get("VLLM_MODEL", match.model_served)),
            vllm_base_url=self._coerce_http_url(
                os.environ.get("VLLM_BASE_URL", match.vllm_base_url),
                "VLLM_BASE_URL or gateway_nodes[].vllm_backend.base_url",
            ),
            vllm_timeout=self._coerce_positive_float(
                os.environ.get("VLLM_TIMEOUT", match.vllm_timeout),
                "VLLM_TIMEOUT or gateway_nodes[].vllm_backend.timeout",
            ),
            max_init_workers=self._coerce_positive_int(
                os.environ.get("GATEWAY_MAX_INIT_WORKERS", match.max_init_workers),
                "GATEWAY_MAX_INIT_WORKERS or gateway_nodes[].max_init_workers",
            ),
            max_run_workers=self._coerce_positive_int(
                os.environ.get("GATEWAY_MAX_RUN_WORKERS", match.max_run_workers),
                "GATEWAY_MAX_RUN_WORKERS or gateway_nodes[].max_run_workers",
            ),
            max_postrun_workers=self._coerce_positive_int(
                os.environ.get("GATEWAY_MAX_POSTRUN_WORKERS", match.max_postrun_workers),
                "GATEWAY_MAX_POSTRUN_WORKERS or gateway_nodes[].max_postrun_workers",
            ),
            ready_buffer_target=self._coerce_positive_int(
                os.environ.get("GATEWAY_READY_BUFFER_TARGET", match.ready_buffer_target),
                "GATEWAY_READY_BUFFER_TARGET or gateway_nodes[].ready_buffer_target",
            ),
            default_runtime=match.default_runtime,
        )

    @property
    def vllm_base_url(self) -> str:
        return self.selected_gateway_node.vllm_base_url

    @property
    def vllm_timeout(self) -> float:
        return self.selected_gateway_node.vllm_timeout

    @property
    def host(self) -> str:
        return self.selected_gateway_node.host

    @property
    def port(self) -> int:
        return self.selected_gateway_node.port

    @property
    def public_url(self) -> str:
        return self.selected_gateway_node.public_url

    @property
    def model_served(self) -> str:
        """The model name served by the selected gateway node backend."""
        return self.selected_gateway_node.model_served

    @property
    def node_id(self) -> str:
        return self.selected_gateway_node.id

    @property
    def max_run_workers(self) -> int:
        return self.selected_gateway_node.max_run_workers

    @property
    def rollout_server_url(self) -> str | None:
        value = (
            os.environ.get("ROLLOUT_SERVER_URL")
            or self._data.get("rollout_server_url")
            or self._data.get("node", {}).get("rollout_server_url")
        )
        if not value:
            return None
        return self._coerce_http_url(
            value,
            "ROLLOUT_SERVER_URL or rollout_server_url",
        )

    @property
    def heartbeat_interval_seconds(self) -> int:
        return self._coerce_positive_int(
            os.environ.get(
                "GATEWAY_HEARTBEAT_INTERVAL_SECONDS",
                self._data.get("heartbeat_interval_seconds")
                or self._data.get("node", {}).get("heartbeat_interval_seconds", 30),
            ),
            "GATEWAY_HEARTBEAT_INTERVAL_SECONDS or heartbeat_interval_seconds",
        )
