"""Rollout control-plane client for node self-registration and heartbeats."""

from __future__ import annotations

import asyncio
import logging

import httpx

from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import NodeHeartbeatRequest, NodeRegistrationRequest

logger = logging.getLogger(__name__)


class RolloutControlClient:
    """Register this gateway node with the rollout server and keep heartbeats flowing."""

    def __init__(
        self,
        *,
        rollout_server_url: str,
        node_id: str,
        gateway_url: str,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
        ready_buffer_target: int,
        heartbeat_interval_seconds: int,
        node_manager: GatewayNodeManager,
    ) -> None:
        self.rollout_server_url = rollout_server_url.rstrip("/")
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.ready_buffer_target = ready_buffer_target
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.node_manager = node_manager
        self._client = httpx.AsyncClient(base_url=self.rollout_server_url, timeout=15.0)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._register()
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._client.aclose()

    async def _register(self) -> None:
        try:
            response = await self._client.post(
                "/nodes/register",
                json=NodeRegistrationRequest(
                    node_id=self.node_id,
                    gateway_url=self.gateway_url,
                    max_init_workers=self.max_init_workers,
                    max_run_workers=self.max_run_workers,
                    max_postrun_workers=self.max_postrun_workers,
                    ready_buffer_target=self.ready_buffer_target,
                    heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                ).model_dump(mode="json"),
            )
            response.raise_for_status()
        except Exception:
            logger.warning("Node registration failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_seconds)
            try:
                metrics = await self.node_manager.stage_metrics()
                response = await self._client.post(
                    f"/nodes/{self.node_id}/heartbeat",
                    json=NodeHeartbeatRequest(metrics=metrics).model_dump(mode="json"),
                )
                if response.status_code == 404:
                    await self._register()
                    continue
                response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Node heartbeat failed", exc_info=True)
