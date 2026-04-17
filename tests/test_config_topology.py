"""Smoke tests for topology config parsing and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from polar.config.topology import GatewayNodeConfig, TopologyConfig


BASE_YAML = """
rollout:
  host: 127.0.0.1
  port: 8080
  public_url: http://127.0.0.1:8080

gateway:
  heartbeat_interval_seconds: 30
  nodes:
    - id: node-a
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      max_init_workers: 4
      max_run_workers: 4
      max_postrun_workers: 4
      model_served: test/model
      sglang:
        base_url: http://127.0.0.1:9000
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "topology.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_happy_path(tmp_path: Path) -> None:
    topology = TopologyConfig.load(_write(tmp_path, BASE_YAML))
    assert topology.rollout.public_url == "http://127.0.0.1:8080"
    assert topology.gateway.rollout_server_url == "http://127.0.0.1:8080"
    assert len(topology.gateway.nodes) == 1
    node = topology.gateway.nodes[0]
    assert node.id == "node-a"
    assert node.public_url == "http://127.0.0.1:8100"
    assert node.sglang_base_url == "http://127.0.0.1:9000"
    assert node.max_init_workers == 4


def test_worker_knobs_are_three(tmp_path: Path) -> None:
    topology = TopologyConfig.load(_write(tmp_path, BASE_YAML))
    node = topology.gateway.nodes[0]
    # Only init/run/postrun are user-visible knobs; the old prewarm/ready knobs are gone.
    assert not hasattr(node, "ready_buffer_target")
    assert not hasattr(node, "max_eval_prewarm_workers")


@pytest.mark.parametrize("legacy_key", ["ready_buffer_target: 4", "max_eval_prewarm_workers: 4"])
def test_legacy_worker_knobs_rejected(tmp_path: Path, legacy_key: str) -> None:
    yaml = BASE_YAML.replace(
        "      max_postrun_workers: 4",
        f"      max_postrun_workers: 4\n      {legacy_key}",
    )
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_legacy_sglang_timeout_rejected(tmp_path: Path) -> None:
    yaml = BASE_YAML.replace(
        "        base_url: http://127.0.0.1:9000",
        "        base_url: http://127.0.0.1:9000\n        timeout: 300",
    )
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_public_url_defaulted_when_missing(tmp_path: Path) -> None:
    yaml = BASE_YAML.replace("      public_url: http://127.0.0.1:8100\n", "")
    topology = TopologyConfig.load(_write(tmp_path, yaml))
    node = topology.gateway.nodes[0]
    assert node.public_url == "http://127.0.0.1:8100"


def test_port_bounds(tmp_path: Path) -> None:
    yaml = BASE_YAML.replace("      port: 8100", "      port: 70000")
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_worker_counts_positive(tmp_path: Path) -> None:
    yaml = BASE_YAML.replace("      max_run_workers: 4", "      max_run_workers: 0")
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_http_url_required_for_public_url(tmp_path: Path) -> None:
    yaml = BASE_YAML.replace(
        "      public_url: http://127.0.0.1:8100",
        "      public_url: not-a-url",
    )
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_duplicate_node_ids_rejected(tmp_path: Path) -> None:
    yaml = (
        BASE_YAML
        + """
    - id: node-a
      host: 127.0.0.1
      port: 8101
      public_url: http://127.0.0.1:8101
      max_init_workers: 4
      max_run_workers: 4
      max_postrun_workers: 4
      sglang:
        base_url: http://127.0.0.1:9001
"""
    )
    with pytest.raises(ValidationError):
        TopologyConfig.load(_write(tmp_path, yaml))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TopologyConfig.load(tmp_path / "does-not-exist.yaml")


def test_bootstrap_nodes_reflect_new_surface(tmp_path: Path) -> None:
    topology = TopologyConfig.load(_write(tmp_path, BASE_YAML))
    boot = topology.bootstrap_nodes[0]
    # Legacy wire fields must not appear.
    assert "ready_buffer_target" not in boot
    assert "max_eval_prewarm_workers" not in boot
    assert boot["node_id"] == "node-a"
    assert boot["gateway_url"] == "http://127.0.0.1:8100"


def test_gateway_node_config_frozen() -> None:
    node = GatewayNodeConfig(
        host="127.0.0.1",
        port=8100,
        public_url="http://127.0.0.1:8100",
        max_init_workers=4,
        max_run_workers=4,
        max_postrun_workers=4,
    )
    with pytest.raises(ValidationError):
        node.port = 8200  # type: ignore[misc]
