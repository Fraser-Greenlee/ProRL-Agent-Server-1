import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from pydantic import BaseModel


def kill_all_singularity_jobs():
    try:
        # List all singularity-related processes
        result = subprocess.run(
            ['pgrep', '-f', 'apptainer'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                subprocess.run(['kill', '-9', pid])
                print(f'Killed Singularity process with PID: {pid}')
    except subprocess.CalledProcessError:
        print('No Singularity processes found.')

    try:
        # List all openhands processes
        result = subprocess.run(
            ['pgrep', '-f', 'openhands'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                subprocess.run(['kill', '-9', pid])
                print(f'Killed Openhands process with PID: {pid}')
    except subprocess.CalledProcessError:
        print('No Openhands processes found.')


# Custom exceptions
class ServerNotRunningError(Exception):
    pass


class NoLLMServerError(Exception):
    pass


class JobTimeoutError(Exception):
    pass


class ProcessRequest(BaseModel):
    instance: dict[str, Any]
    sampling_params: dict[str, Any]


class LLMServerRequest(BaseModel):
    address: str


async def cleanup_timed_out_job(server, job_id: str | None = None):
    """Clean up a job that has timed out or failed"""
    if job_id is None:
        return

    if job_id in server._job_details:
        job_details = server._job_details[job_id]

        # Set the event to unblock any waiting threads
        if job_details.event:
            job_details.event.set()

        # Remove from active jobs sets
        if job_id in server._active_init_jobs:
            server._active_init_jobs.remove(job_id)
        if job_id in server._active_run_jobs:
            server._active_run_jobs.remove(job_id)

        # Close runtime if it exists
        if job_details.runtime:
            try:
                job_details.runtime.close()
            except Exception:
                pass

        # Remove from job details
        if job_id in server._job_details:
            del server._job_details[job_id]


async def process_with_timeout(
    server,
    instance: pd.Series,
    sampling_params: dict[str, Any],
    timeout: float,
    thread_pool: ThreadPoolExecutor | None = None,
    job_id: str | None = None,
):
    if job_id is None:
        job_id = server.get_unique_id(instance)
    try:
        # Create a future for the process call

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            thread_pool, lambda: server.process(instance, sampling_params, job_id)
        )

        # Wait for the future with timeout
        result = await asyncio.wait_for(future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        # Clean up the timed-out job
        await cleanup_timed_out_job(server, job_id)
        raise JobTimeoutError(f'Job {job_id} timed out after {timeout} seconds')
    except Exception:
        # Clean up on any other error
        await cleanup_timed_out_job(server, job_id)
        raise
