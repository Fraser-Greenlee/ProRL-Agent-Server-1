"""Configuration loading for the rollout service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


class RolloutConfig:
    """Load rollout service settings from YAML plus environment overrides."""

    def __init__(self, config_path: str | None = None, *, require_existing: bool = False):
        self._data = self._load_data(config_path, require_existing=require_existing)
        self._validate()

    @classmethod
    def from_environment(cls) -> "RolloutConfig":
        config_path = os.environ.get("CONFIG_PATH")
        if config_path is None:
            return cls("config.yaml")
        return cls(config_path, require_existing=True)

    @staticmethod
    def _load_data(config_path: str | None, *, require_existing: bool) -> dict[str, Any]:
        if not config_path:
            if require_existing:
                raise ValueError("Rollout config path cannot be empty")
            return {}

        path = Path(config_path)
        if not path.exists():
            if require_existing:
                raise FileNotFoundError(f"Rollout config file not found: {config_path}")
            return {}
        if not path.is_file():
            raise ValueError(f"Rollout config path is not a file: {config_path}")

        try:
            with path.open() as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid rollout YAML in {config_path}: {exc}") from exc

        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Rollout config {config_path} must contain a top-level mapping"
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

    @staticmethod
    def _coerce_non_negative_float(value: object, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number") from exc
        if parsed < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0")
        return parsed

    @classmethod
    def _coerce_http_url(cls, value: object, field_name: str) -> str:
        text = cls._require_non_empty_string(value, field_name)
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{field_name} must be an http:// or https:// URL")
        return text.rstrip("/")

    def _rollout_section(self) -> dict[str, Any]:
        rollout = self._data.get("rollout", {})
        if rollout is None:
            return {}
        if not isinstance(rollout, dict):
            raise ValueError("rollout must be a mapping")
        return rollout

    def _validate(self) -> None:
        rollout = self._rollout_section()
        if "host" in rollout:
            self._require_non_empty_string(rollout["host"], "rollout.host")
        if "port" in rollout:
            self._coerce_port(rollout["port"], "rollout.port")
        if "public_url" in rollout and rollout["public_url"] is not None:
            self._coerce_http_url(rollout["public_url"], "rollout.public_url")
        if "save_dir" in rollout and rollout["save_dir"] is not None:
            self._require_non_empty_string(rollout["save_dir"], "rollout.save_dir")
        if "results_dir" in rollout and rollout["results_dir"] is not None:
            self._require_non_empty_string(rollout["results_dir"], "rollout.results_dir")
        if "dispatch_poll_interval_seconds" in rollout:
            self._coerce_positive_float(
                rollout["dispatch_poll_interval_seconds"],
                "rollout.dispatch_poll_interval_seconds",
            )
        if "callback_grace_seconds" in rollout:
            self._coerce_non_negative_float(
                rollout["callback_grace_seconds"],
                "rollout.callback_grace_seconds",
            )

        nodes = rollout.get("nodes", [])
        if nodes is None or not isinstance(nodes, list):
            raise ValueError("rollout.nodes must be a list")

        seen_node_ids: set[str] = set()
        for index, entry in enumerate(nodes):
            if not isinstance(entry, dict):
                raise ValueError(f"rollout.nodes[{index}] must be a mapping")

            node_id = self._require_non_empty_string(
                entry.get("node_id"),
                f"rollout.nodes[{index}].node_id",
            )
            if node_id in seen_node_ids:
                raise ValueError(f"Duplicate rollout.nodes node_id: {node_id}")
            seen_node_ids.add(node_id)

            self._coerce_http_url(
                entry.get("gateway_url"),
                f"rollout.nodes[{index}].gateway_url",
            )
            self._coerce_positive_int(
                entry.get("max_init_workers", 4),
                f"rollout.nodes[{index}].max_init_workers",
            )
            self._coerce_positive_int(
                entry.get("max_run_workers", entry.get("capacity", 1)),
                f"rollout.nodes[{index}].max_run_workers",
            )
            self._coerce_positive_int(
                entry.get("max_postrun_workers", 4),
                f"rollout.nodes[{index}].max_postrun_workers",
            )
            self._coerce_positive_int(
                entry.get("ready_buffer_target", entry.get("max_run_workers", entry.get("capacity", 1))),
                f"rollout.nodes[{index}].ready_buffer_target",
            )
            self._coerce_positive_int(
                entry.get("heartbeat_interval_seconds", 30),
                f"rollout.nodes[{index}].heartbeat_interval_seconds",
            )

    @property
    def host(self) -> str:
        return self._require_non_empty_string(
            os.environ.get(
                "ROLLOUT_HOST",
                self._rollout_section().get("host", "0.0.0.0"),
            ),
            "ROLLOUT_HOST or rollout.host",
        )

    @property
    def port(self) -> int:
        return self._coerce_port(
            os.environ.get(
                "ROLLOUT_PORT",
                self._rollout_section().get("port", 8082),
            ),
            "ROLLOUT_PORT or rollout.port",
        )

    @property
    def public_url(self) -> str:
        configured = os.environ.get("ROLLOUT_PUBLIC_URL") or self._rollout_section().get(
            "public_url"
        )
        if configured:
            return self._coerce_http_url(
                configured,
                "ROLLOUT_PUBLIC_URL or rollout.public_url",
            )

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
            rollout = self._rollout_section()
            configured = rollout.get("save_dir")
            if configured is None:
                configured = rollout.get("results_dir")
        if configured is None:
            return None
        return self._require_non_empty_string(
            configured,
            "ROLLOUT_SAVE_DIR or ROLLOUT_RESULTS_DIR or rollout.save_dir",
        )

    @property
    def results_dir(self) -> str | None:
        return self.save_dir

    @property
    def dispatch_poll_interval_seconds(self) -> float:
        return self._coerce_positive_float(
            os.environ.get(
                "ROLLOUT_DISPATCH_POLL_INTERVAL_SECONDS",
                self._rollout_section().get("dispatch_poll_interval_seconds", 1.0),
            ),
            (
                "ROLLOUT_DISPATCH_POLL_INTERVAL_SECONDS or "
                "rollout.dispatch_poll_interval_seconds"
            ),
        )

    @property
    def callback_grace_seconds(self) -> float:
        return self._coerce_non_negative_float(
            os.environ.get(
                "ROLLOUT_CALLBACK_GRACE_SECONDS",
                self._rollout_section().get("callback_grace_seconds", 5.0),
            ),
            "ROLLOUT_CALLBACK_GRACE_SECONDS or rollout.callback_grace_seconds",
        )

    @property
    def bootstrap_nodes(self) -> list[dict[str, object]]:
        configured = self._rollout_section().get("nodes", [])
        nodes: list[dict[str, object]] = []
        for index, entry in enumerate(configured):
            nodes.append(
                {
                    "node_id": self._require_non_empty_string(
                        entry.get("node_id"),
                        f"rollout.nodes[{index}].node_id",
                    ),
                    "gateway_url": self._coerce_http_url(
                        entry.get("gateway_url"),
                        f"rollout.nodes[{index}].gateway_url",
                    ),
                    "max_init_workers": self._coerce_positive_int(
                        entry.get("max_init_workers", 4),
                        f"rollout.nodes[{index}].max_init_workers",
                    ),
                    "max_run_workers": self._coerce_positive_int(
                        entry.get("max_run_workers", entry.get("capacity", 1)),
                        f"rollout.nodes[{index}].max_run_workers",
                    ),
                    "max_postrun_workers": self._coerce_positive_int(
                        entry.get("max_postrun_workers", 4),
                        f"rollout.nodes[{index}].max_postrun_workers",
                    ),
                    "ready_buffer_target": self._coerce_positive_int(
                        entry.get("ready_buffer_target", entry.get("max_run_workers", entry.get("capacity", 1))),
                        f"rollout.nodes[{index}].ready_buffer_target",
                    ),
                    "heartbeat_interval_seconds": self._coerce_positive_int(
                        entry.get("heartbeat_interval_seconds", 30),
                        f"rollout.nodes[{index}].heartbeat_interval_seconds",
                    ),
                }
            )
        return nodes
