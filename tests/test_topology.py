from pathlib import Path

import pytest

from polar.config import TopologyConfig


def test_topology_loads_calculator_example():
    topology = TopologyConfig.load(
        Path("examples/calculator/shared/topology.yaml")
    )

    assert topology.rollout.public_url == "http://127.0.0.1:8080"
    assert [node.id for node in topology.gateway.nodes] == [
        "localhost-node-01",
        "localhost-node-02",
    ]
    assert topology.select_gateway_node("localhost-node-02").vllm_base_url == "http://127.0.0.1:8001"


def test_topology_requires_node_id_when_multiple_nodes():
    topology = TopologyConfig.load(
        Path("examples/calculator/shared/topology.yaml")
    )

    with pytest.raises(ValueError, match="--node-id"):
        topology.select_gateway_node()
