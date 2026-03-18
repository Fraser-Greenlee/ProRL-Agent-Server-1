"""Gateway-node registration and least-loaded scheduling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import threading

from rollout.models import GatewayNodeInfo, NodeRegistrationRequest


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class GatewayNode:
    node_id: str
    gateway_url: str
    capacity: int
    active_sessions: int
    healthy: bool
    last_heartbeat: datetime
    heartbeat_interval_seconds: int
    draining: bool = False

    def to_model(self) -> GatewayNodeInfo:
        return GatewayNodeInfo(
            node_id=self.node_id,
            gateway_url=self.gateway_url,
            capacity=self.capacity,
            active_sessions=self.active_sessions,
            healthy=self.healthy,
            draining=self.draining,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            last_heartbeat=self.last_heartbeat,
        )


class NodeScheduler:
    """Track registered nodes and assign sessions to the least-loaded healthy node."""

    def __init__(
        self,
        *,
        bootstrap_nodes: list[dict[str, object]] | None = None,
        stale_factor: float = 2.5,
    ) -> None:
        self._nodes: dict[str, GatewayNode] = {}
        self._lock = threading.RLock()
        self._stale_factor = stale_factor

        for node in bootstrap_nodes or []:
            node_id = str(node["node_id"])
            self._nodes[node_id] = GatewayNode(
                node_id=node_id,
                gateway_url=str(node["gateway_url"]).rstrip("/"),
                capacity=max(1, int(node.get("capacity", 1))),
                active_sessions=0,
                healthy=False,
                last_heartbeat=_utcnow(),
                heartbeat_interval_seconds=max(1, int(node.get("heartbeat_interval_seconds", 30))),
            )

    def register_node(self, request: NodeRegistrationRequest) -> GatewayNodeInfo:
        now = _utcnow()
        with self._lock:
            existing = self._nodes.get(request.node_id)
            draining = existing.draining if existing is not None else False
            self._nodes[request.node_id] = GatewayNode(
                node_id=request.node_id,
                gateway_url=request.gateway_url.rstrip("/"),
                capacity=request.capacity,
                active_sessions=existing.active_sessions if existing is not None else 0,
                healthy=True,
                last_heartbeat=now,
                heartbeat_interval_seconds=request.heartbeat_interval_seconds,
                draining=draining,
            )
            return self._nodes[request.node_id].to_model()

    def heartbeat(self, node_id: str, *, active_sessions: int | None = None) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            node.last_heartbeat = _utcnow()
            node.healthy = True
            if active_sessions is not None:
                node.active_sessions = max(0, active_sessions)
            return node.to_model()

    def acquire_node(self) -> GatewayNodeInfo | None:
        with self._lock:
            self._refresh_health_locked()
            candidates = [
                node
                for node in self._nodes.values()
                if node.healthy and not node.draining and node.active_sessions < node.capacity
            ]
            if not candidates:
                return None

            selected = min(
                candidates,
                key=lambda node: (
                    node.active_sessions / node.capacity,
                    node.active_sessions,
                    node.node_id,
                ),
            )
            selected.active_sessions += 1
            return selected.to_model()

    def release_session(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.active_sessions = max(0, node.active_sessions - 1)
            snapshot = node.to_model()
            if node.draining and node.active_sessions == 0:
                self._nodes.pop(node_id, None)
            return snapshot

    def mark_unhealthy(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.healthy = False
            return node.to_model()

    def drain_node(self, node_id: str) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            node.draining = True
            snapshot = node.to_model()
            if node.active_sessions == 0:
                self._nodes.pop(node_id, None)
            return snapshot

    def get_node(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            self._refresh_health_locked()
            node = self._nodes.get(node_id)
            return None if node is None else node.to_model()

    def list_nodes(self) -> list[GatewayNodeInfo]:
        with self._lock:
            self._refresh_health_locked()
            return [self._copy_node(node).to_model() for node in sorted(self._nodes.values(), key=lambda item: item.node_id)]

    def stats(self) -> dict[str, object]:
        with self._lock:
            self._refresh_health_locked()
            return {
                "nodes": [self._copy_node(node).to_model().model_dump(mode="json") for node in sorted(self._nodes.values(), key=lambda item: item.node_id)],
            }

    def _refresh_health_locked(self) -> None:
        now = _utcnow()
        for node in self._nodes.values():
            timeout_seconds = max(1.0, node.heartbeat_interval_seconds * self._stale_factor)
            node.healthy = (now - node.last_heartbeat).total_seconds() <= timeout_seconds

    def _require_node_locked(self, node_id: str) -> GatewayNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown gateway node: {node_id}")
        return node

    @staticmethod
    def _copy_node(node: GatewayNode) -> GatewayNode:
        return replace(node)
