"""Configuration loading from YAML + environment variable overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import os
import socket
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class GatewayNodeConfig:
    id: str
    host: str
    port: int
    public_url: str
    capacity: int
    model_served: str
    vllm_base_url: str
    vllm_timeout: float


class Config:
    def __init__(self, config_path: str | None = None):
        self._data: dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self._data = yaml.safe_load(f) or {}

    @staticmethod
    def _default_public_url(host: str, port: int) -> str:
        public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{public_host}:{port}"

    def _legacy_gateway_node_entry(self) -> dict[str, Any]:
        return {
            "id": self._data.get("node", {}).get("id", socket.gethostname()),
            "host": self._data.get("server", {}).get("host", "0.0.0.0"),
            "port": self._data.get("server", {}).get("port", 8080),
            "public_url": self._data.get("server", {}).get("public_url"),
            "capacity": self._data.get("node", {}).get("capacity", 1),
            "model_served": self._data.get("model_served", ""),
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
        for index, entry in enumerate(configured):
            if not isinstance(entry, dict):
                raise ValueError(f"gateway_nodes[{index}] must be a mapping")

            node_id = str(entry.get("id") or socket.gethostname())
            host = str(entry.get("host", "0.0.0.0"))
            port = int(entry.get("port", 8080))
            configured_public_url = entry.get("public_url")
            public_url = str(
                configured_public_url or self._default_public_url(host, port)
            ).rstrip("/")
            capacity = max(1, int(entry.get("capacity", 1)))

            backend = entry.get("vllm_backend", {})
            if backend is None:
                backend = {}
            if not isinstance(backend, dict):
                raise ValueError(f"gateway_nodes[{index}].vllm_backend must be a mapping")

            nodes.append(
                GatewayNodeConfig(
                    id=node_id,
                    host=host,
                    port=port,
                    public_url=public_url,
                    capacity=capacity,
                    model_served=str(entry.get("model_served", self._data.get("model_served", ""))),
                    vllm_base_url=str(
                        backend.get("base_url", "http://localhost:8000")
                    ).rstrip("/"),
                    vllm_timeout=float(backend.get("timeout", 300)),
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

        host = os.environ.get("PROXY_HOST", match.host)
        port = int(os.environ.get("PROXY_PORT", match.port))
        configured_public_url = os.environ.get("PROXY_PUBLIC_URL")
        if configured_public_url:
            public_url = str(configured_public_url).rstrip("/")
        elif "PROXY_HOST" in os.environ or "PROXY_PORT" in os.environ:
            public_url = self._default_public_url(host, port)
        else:
            public_url = match.public_url

        return GatewayNodeConfig(
            id=match.id,
            host=host,
            port=port,
            public_url=public_url,
            capacity=int(os.environ.get("GATEWAY_NODE_CAPACITY", match.capacity)),
            model_served=str(os.environ.get("VLLM_MODEL", match.model_served)),
            vllm_base_url=str(
                os.environ.get("VLLM_BASE_URL", match.vllm_base_url)
            ).rstrip("/"),
            vllm_timeout=float(os.environ.get("VLLM_TIMEOUT", match.vllm_timeout)),
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
    def node_capacity(self) -> int:
        return self.selected_gateway_node.capacity

    @property
    def rollout_server_url(self) -> str | None:
        value = (
            os.environ.get("ROLLOUT_SERVER_URL")
            or self._data.get("rollout_server_url")
            or self._data.get("node", {}).get("rollout_server_url")
        )
        if not value:
            return None
        return str(value).rstrip("/")

    @property
    def heartbeat_interval_seconds(self) -> int:
        return int(
            os.environ.get(
                "GATEWAY_HEARTBEAT_INTERVAL_SECONDS",
                self._data.get("heartbeat_interval_seconds")
                or self._data.get("node", {}).get("heartbeat_interval_seconds", 30),
            )
        )
