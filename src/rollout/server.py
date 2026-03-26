"""FastAPI server for rollout orchestration."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rollout.balancer import NodeScheduler
from rollout.config import RolloutConfig
from rollout.manager import RolloutManager
from rollout.models import (
    GatewayNodeInfo,
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    SessionResult,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from rollout.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

config = RolloutConfig.from_environment()
scheduler = NodeScheduler(bootstrap_nodes=config.bootstrap_nodes)
pipeline = Pipeline(
    callback_url=f"{config.public_url}/callbacks/session_result",
    save_dir=config.save_dir,
    scheduler=scheduler,
    dispatch_poll_interval_seconds=config.dispatch_poll_interval_seconds,
    callback_grace_seconds=config.callback_grace_seconds,
)
manager = RolloutManager(pipeline=pipeline, scheduler=scheduler)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    await pipeline.start()
    try:
        yield
    finally:
        await pipeline.close()


app = FastAPI(title="Rollout Server", version="0.2.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "nodes": len(scheduler.list_nodes())}


@app.post("/rollout/task", response_model=TaskResult)
async def submit_task(request: TaskRequest):
    try:
        return await manager.execute_task(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/rollout/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    task = manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/rollout/status")
async def rollout_status():
    return manager.status()


@app.post("/nodes/register", response_model=GatewayNodeInfo)
async def register_node(request: NodeRegistrationRequest):
    return scheduler.register_node(request)


@app.post("/nodes/{node_id}/heartbeat", response_model=GatewayNodeInfo)
async def node_heartbeat(node_id: str, request: NodeHeartbeatRequest):
    try:
        return scheduler.heartbeat(node_id, metrics=request.metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/nodes", response_model=list[GatewayNodeInfo])
async def list_nodes():
    return scheduler.list_nodes()


@app.get("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def get_node(node_id: str):
    node = scheduler.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.delete("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def drain_node(node_id: str):
    try:
        return scheduler.drain_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/callbacks/session_result")
async def session_result_callback(result: SessionResult):
    await pipeline.accept_callback_result(result)
    return {"status": "accepted"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rollout.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
