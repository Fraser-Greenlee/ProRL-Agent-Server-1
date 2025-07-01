import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from openhands.nvidia.async_server import OpenHandsServer
from openhands.nvidia.registry import FunctionNotRegisteredError
from openhands.nvidia.utils import (
    JobTimeoutError,
    LLMServerRequest,
    NoLLMServerError,
    ProcessRequest,
    ServerNotRunningError,
    kill_all_singularity_jobs,
    process_with_timeout,
)
from openhands.nvidia.logger import nvidia_logger as logger

app = FastAPI(title='OpenHands Async Server API')

# Global server instance
server = None
# Thread pool for submitting jobs
thread_pool = None

# Global timeout configuration (in seconds)
DEFAULT_TIMEOUT = 300.0  # 5 minutes
global_timeout = DEFAULT_TIMEOUT


def init_server(
    max_init_workers: int = 6,
    max_run_workers: int = 5,
    timeout: float = DEFAULT_TIMEOUT,
    allow_skip_eval: bool = True,
):
    logger.info(
        f'Initializing server with max_init_workers={max_init_workers}, max_run_workers={max_run_workers}, timeout={timeout}'
    )
    if allow_skip_eval:
        logger.info(
            'Allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=False for testing.'
        )
    else:
        logger.info(
            'Not allowing skipping evaluation if git_patch is None or empty. Please set allow_skip_eval=True for production.'
        )
    global server, global_timeout, thread_pool
    server = OpenHandsServer(
        llm_server_addresses=[],
        max_init_workers=max_init_workers,
        max_run_workers=max_run_workers,
        allow_skip_eval=allow_skip_eval,
    )
    global_timeout = timeout
    cpu_count = os.cpu_count()
    if cpu_count is None:
        # Fallback to a reasonable default if CPU count is undetermined
        thread_pool_count = max_init_workers
    else:
        thread_pool_count = min(max_init_workers, cpu_count - 64)
    logger.info(f'Using {thread_pool_count} threads for the thread pool')
    thread_pool = ThreadPoolExecutor(max_workers=thread_pool_count)


@app.exception_handler(ServerNotRunningError)
async def server_not_running_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={'detail': 'Server is not running. Please start the server first.'},
    )


@app.exception_handler(NoLLMServerError)
async def no_llm_server_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            'detail': 'No LLM server addresses configured. Please add at least one LLM server address.'
        },
    )


@app.exception_handler(JobTimeoutError)
async def job_timeout_handler(request, exc):
    return JSONResponse(status_code=504, content={'detail': str(exc)})


@app.exception_handler(FunctionNotRegisteredError)
async def function_not_registered_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            'detail': f'Invalid dataset type or function not registered: {str(exc)}'
        },
    )


@app.post('/start')
async def start_server():
    kill_all_singularity_jobs()
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if server._server_running:
        logger.warning('Server is already running. But user requested to start.')
        raise HTTPException(status_code=400, detail='Server is already running')

    try:
        server.start()
        return {'status': 'Server started successfully'}
    except Exception as e:
        logger.error(f'Failed to start server: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to start server: {str(e)}')


@app.post('/stop')
async def stop_server():
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to stop.')
        raise ServerNotRunningError()
    try:
        # Run the stop operation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.stop)
        kill_all_singularity_jobs()
        return {'status': 'Server stopped successfully'}
    except Exception as e:
        logger.warning(f'Failed to stop server: {str(e)}. Force kill all singularity jobs.')
        kill_all_singularity_jobs()
        return {'status': 'Force killed all singularity jobs.'}


@app.get('/status')
async def get_status():
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to get status.')
        raise ServerNotRunningError()

    try:
        return server.status()
    except Exception as e:
        logger.error(f'Server probably not running. Failed to get server status: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Server probably not running. Failed to get server status: {str(e)}',
        )


@app.post('/add_llm_server')
async def add_llm_server(request: LLMServerRequest):
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    try:
        server.add_llm_server_address(request.address)
        return {'status': f'Added LLM server address: {request.address}'}
    except Exception as e:
        logger.error(f'Failed to add LLM server: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to add LLM server: {str(e)}'
        )


@app.post('/clear_llm_server')
async def clear_llm_server():
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    try:
        server.clear_llm_server_addresses()
        return {'status': 'Cleared all LLM server addresses'}
    except Exception as e:
        logger.error(f'Failed to clear LLM servers: {str(e)}')
        raise HTTPException(
            status_code=500, detail=f'Failed to clear LLM servers: {str(e)}'
        )


@app.post('/process')
async def process(request: ProcessRequest):
    global server
    if server is None:
        logger.error('Server is not initialized. This should not happen.')
        raise HTTPException(
            status_code=500, detail='Server is not initialized. This should not happen.'
        )
    if not server._server_running:
        logger.warning('Server is not running. But user requested to process.')
        raise ServerNotRunningError()

    if len(server.weighted_addresses) == 0:
        logger.error('No LLM server addresses configured. Please add at least one LLM server address.')
        raise NoLLMServerError()

    # Convert instance dict to pandas Series
    try:
        instance = request.instance
    except Exception as e:
        logger.error(f'Invalid instance data: {str(e)}')
        raise HTTPException(status_code=400, detail=f'Invalid instance data: {str(e)}')

    result = await process_with_timeout(
        server, instance, request.sampling_params, global_timeout, thread_pool
    )
    return result


def start_api_server(host: str = '0.0.0.0', port: int = 8000):
    uvicorn.run(app, host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='OpenHands Async Server API')
    parser.add_argument(
        '--max-init-workers',
        type=int,
        default=64,
        help='Maximum number of initialization workers (default: 64)',
    )
    parser.add_argument(
        '--max-run-workers',
        type=int,
        default=64,
        help='Maximum number of run workers (default: 64)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f'Global timeout for job processing in seconds (default: {DEFAULT_TIMEOUT})',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8006,
        help='Port to bind the server to (default: 8006)',
    )
    parser.add_argument(
        '--allow-skip-eval',
        type=bool,
        default=True,
        help='Allow skipping evaluation if git_patch is None or empty. Set to False for testing (default: True).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_server(
        max_init_workers=args.max_init_workers,
        max_run_workers=args.max_run_workers,
        timeout=args.timeout,
        allow_skip_eval=args.allow_skip_eval,
    )
    start_api_server(host=args.host, port=args.port)
