"""Configuration loading for the rollout service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class RolloutConfig:
    """Load rollout service settings from YAML plus environment overrides."""

    def __init__(self, config_path: str | None = None):
        self._data: dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as handle:
                self._data = yaml.safe_load(handle) or {}

    @property
    def host(self) -> str:
        return os.environ.get(
            "ROLLOUT_HOST",
            self._data.get("rollout", {}).get("host", "0.0.0.0"),
        )

    @property
    def port(self) -> int:
        return int(
            os.environ.get(
                "ROLLOUT_PORT",
                self._data.get("rollout", {}).get("port", 8082),
            )
        )

    @property
    def public_url(self) -> str:
        configured = os.environ.get("ROLLOUT_PUBLIC_URL") or self._data.get("rollout", {}).get("public_url")
        if configured:
            return str(configured).rstrip("/")

        host = self.host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def save_dir(self) -> str | None:
        configured = os.environ.get("ROLLOUT_SAVE_DIR")
        if configured is None:
            configured = os.environ.get("ROLLOUT_RESULTS_DIR")
        if configured is None:
            rollout = self._data.get("rollout", {})
            configured = rollout.get("save_dir")
            if configured is None:
                configured = rollout.get("results_dir")
        if configured is None:
            return None
        value = str(configured).strip()
        return value or None

    @property
    def results_dir(self) -> str | None:
        return self.save_dir

    @property
    def dispatch_poll_interval_seconds(self) -> float:
        return float(
            os.environ.get(
                "ROLLOUT_DISPATCH_POLL_INTERVAL_SECONDS",
                self._data.get("rollout", {}).get("dispatch_poll_interval_seconds", 1.0),
            )
        )

    @property
    def callback_grace_seconds(self) -> float:
        return float(
            os.environ.get(
                "ROLLOUT_CALLBACK_GRACE_SECONDS",
                self._data.get("rollout", {}).get("callback_grace_seconds", 5.0),
            )
        )

    @property
    def bootstrap_nodes(self) -> list[dict[str, object]]:
        configured = self._data.get("rollout", {}).get("nodes", [])
        if not isinstance(configured, list):
            return []
        nodes: list[dict[str, object]] = []
        for entry in configured:
            if not isinstance(entry, dict):
                continue
            if "node_id" not in entry or "gateway_url" not in entry:
                continue
            nodes.append(
                {
                    "node_id": str(entry["node_id"]),
                    "gateway_url": str(entry["gateway_url"]).rstrip("/"),
                    "capacity": int(entry.get("capacity", 1)),
                    "heartbeat_interval_seconds": int(entry.get("heartbeat_interval_seconds", 30)),
                }
            )
        return nodes
