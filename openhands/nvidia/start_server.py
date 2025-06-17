from fastapi import FastAPI, HTTPException
import pandas as pd
import uvicorn
import argparse
from openhands.nvidia.async_server import OpenHandsServer
from fastapi.responses import JSONResponse
from openhands.nvidia.utils import ServerNotRunningError, NoLLMServerError, JobTimeoutError, ProcessRequest, LLMServerRequest, process_with_timeout, kill_all_singularity_jobs
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI(title="OpenHands Async Server API")

# Global server instance
server = None
# Thread pool for submitting jobs
thread_pool = None

# Global timeout configuration (in seconds)
DEFAULT_TIMEOUT = 300.0  # 5 minutes
global_timeout = DEFAULT_TIMEOUT

def init_server(max_init_workers: int = 6, max_run_workers: int = 5, timeout: float = DEFAULT_TIMEOUT, allow_skip_eval: bool = True):
    print(f"Initializing server with max_init_workers={max_init_workers}, max_run_workers={max_run_workers}, timeout={timeout}")
    if allow_skip_eval:
        print("Allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=False for testing.")
    else:
        print("Not allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=True for production.")
    global server, global_timeout, thread_pool
    server = OpenHandsServer(
        llm_server_addresses=[],
        max_init_workers=max_init_workers,
        max_run_workers=max_run_workers,
        allow_skip_eval=allow_skip_eval
    )
    global_timeout = timeout
    thread_pool_count = min( max_init_workers, os.cpu_count() - 64)
    print(f"Using {thread_pool_count} threads for the thread pool")
    thread_pool = ThreadPoolExecutor(max_workers=thread_pool_count)

@app.exception_handler(ServerNotRunningError)
async def server_not_running_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Server is not running. Please start the server first."}
    )

@app.exception_handler(NoLLMServerError)
async def no_llm_server_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "No LLM server addresses configured. Please add at least one LLM server address."}
    )

@app.exception_handler(JobTimeoutError)
async def job_timeout_handler(request, exc):
    return JSONResponse(
        status_code=504,
        content={"detail": str(exc)}
    )

@app.post("/start")
async def start_server():
    kill_all_singularity_jobs()
    global server
    if hasattr(server, 'init_queue') and server.init_queue is not None:
        raise HTTPException(status_code=400, detail="Server is already running")
    
    try:
        server.start()
        return {"status": "Server started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

@app.post("/stop")
async def stop_server():
    global server
    if not hasattr(server, 'init_queue') or server.init_queue is None:
        raise ServerNotRunningError()
    try:
        # Run the stop operation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.stop)
        kill_all_singularity_jobs()
        return {"status": "Server stopped successfully"}
    except Exception as e:
        print(f"Failed to stop server: {str(e)}. Force kill all singularity jobs.")
        kill_all_singularity_jobs()
        return {"status": "Force killed all singularity jobs."}

@app.get("/status")
async def get_status():
    global server
    if not hasattr(server, 'init_queue') or server.init_queue is None:
        raise ServerNotRunningError()
    
    try:
        return server.status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server probably not running. Failed to get server status: {str(e)}")

@app.post("/add_llm_server")
async def add_llm_server(request: LLMServerRequest):
    global server
    try:
        server.add_llm_server_address(request.address)
        return {"status": f"Added LLM server address: {request.address}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add LLM server: {str(e)}")


@app.post("/process")
async def process(request: ProcessRequest):
    global server
    if not hasattr(server, 'init_queue') or server.init_queue is None:
        raise ServerNotRunningError()
    
    if len(server.weighted_addresses) == 0:
        raise NoLLMServerError()
    
    # Convert instance dict to pandas Series
    try:
        instance = pd.Series(request.instance)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid instance data: {str(e)}")
    
    result = await process_with_timeout(server, instance, request.sampling_params, global_timeout, thread_pool)
    return result

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

def parse_args():
    parser = argparse.ArgumentParser(description='OpenHands Async Server API')
    parser.add_argument('--max-init-workers', type=int, default=6,
                      help='Maximum number of initialization workers (default: 6)')
    parser.add_argument('--max-run-workers', type=int, default=5,
                      help='Maximum number of run workers (default: 5)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                      help=f'Global timeout for job processing in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                      help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8006,
                      help='Port to bind the server to (default: 8006)')
    parser.add_argument('--allow-skip-eval', type=bool, default=True,
                      help='Allow skipping evaluation if git_patch is None or empty. Set to False for testing (default: True).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    init_server(max_init_workers=args.max_init_workers, max_run_workers=args.max_run_workers, timeout=args.timeout, allow_skip_eval=args.allow_skip_eval)
    start_api_server(host=args.host, port=args.port) 