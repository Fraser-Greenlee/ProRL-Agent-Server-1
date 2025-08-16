# type: ignore
import asyncio
import hashlib
import heapq
import multiprocessing as mp
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, cast

from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import (
    FunctionNotRegisteredError,
    JobDetails,
    get_registered_functions,
    is_registered_handler,
)
from openhands.nvidia.reward import Reward
from openhands.nvidia.timer import (
    PausableTimer,
    TimeoutError,
    phase_context,
    run_with_timeout_awareness,
)
from openhands.nvidia.utils import (
    clear_queue,
    get_instance_id,
    get_singularity_job_pids,
    kill_all_singularity_jobs,
)


@dataclass
class ConcurrencyControl:
    init: mp.Semaphore
    run: mp.Semaphore
    eval: mp.Semaphore

    def __init__(
        self,
        max_init_workers: int,
        max_run_workers: int,
        max_eval_workers: int,
    ):
        self.init = mp.Semaphore(max_init_workers)
        self.run = mp.Semaphore(max_run_workers)
        self.eval = mp.Semaphore(max_eval_workers)

    def release_all(self):
        for _ in range(self.init.get_value()):
            self.init.release()
        for _ in range(self.run.get_value()):
            self.run.release()
        for _ in range(self.eval.get_value()):
            self.eval.release()


class JobType(Enum):
    INIT = 'init'
    RUN = 'run'
    EVAL = 'eval'


class Worker:
    def __init__(
        self,
        job_id: str,
        instance: dict,
        sampling_params: dict,
        timeout: float,
        llm_server_addresses: str,
        allow_skip_eval: bool = True,
        reward_server_ip: list[str] | None = None,
        concurrency_control: Union[ConcurrencyControl, None] = None,
        job_status: Union[dict, None] = None,
        job_status_lock: Union[mp.Lock, None] = None,
    ):
        self.job_id = job_id  # Add job_id attribute for compatibility
        self.instance = instance
        self.sampling_params = sampling_params
        self.timeout = timeout
        self.llm_server_addresses = llm_server_addresses
        self.allow_skip_eval = allow_skip_eval
        if reward_server_ip is not None:
            self.reward = Reward(server_ip=reward_server_ip)
        else:
            self.reward = None
        self.concurrency_control = concurrency_control
        self.job_status = job_status
        self.job_status_lock = job_status_lock

    def create_llm_config(self):
        llm_config = LLMConfig(
            base_url=self.llm_server_addresses, **self.sampling_params
        )
        return llm_config

    def initialize(self):
        job_details = JobDetails()
        job_details.job_id = self.job_id
        job_details.instance = self.instance
        for agent_config_key in job_details.agent_config:
            if agent_config_key in self.sampling_params:
                job_details.agent_config[agent_config_key] = self.sampling_params.pop(
                    agent_config_key
                )
        llm_config = self.create_llm_config()
        job_details.llm_config = llm_config
        job_details.event = threading.Event()

        job_details.timer = PausableTimer(timeout=self.timeout)
        job_details.timer.start()

        self.job_details = job_details

    def _cleanup_job_runtime(self, runtime, job_id: str):
        """Comprehensive cleanup of runtime resources to prevent thread leakage."""

        def close():
            try:
                # 1. Close runtime (handles container processes, plugins, etc.)
                runtime.close()
                # 2. Close event stream and its thread pools
                if hasattr(runtime, 'event_stream') and runtime.event_stream:
                    try:
                        runtime.event_stream.close()
                    except Exception:
                        pass
                # 3. Force cleanup any remaining subprocess-related resources
                time.sleep(0.1)  # Brief pause for cleanup to complete

            except Exception:
                pass
                # Don't re-raise - we want cleanup to continue even if parts fail

        # Run cleanup in background thread, non-blocking
        t = threading.Thread(target=close, daemon=True)
        t.start()

    async def run_step(self, job_type: JobType):
        # Map job types to their respective queues and tracking sets
        job_config = {
            JobType.INIT: {
                'function_type': 'init',
                'exception_type': 'init_exception',
            },
            JobType.RUN: {
                'function_type': 'run',
                'exception_type': 'run_exception',
            },
            JobType.EVAL: {
                'function_type': 'eval',
                'exception_type': 'eval_exception',
            },
        }

        job_config_data = job_config[job_type]
        function_type: str = cast(str, job_config_data['function_type'])
        exception_type: str = cast(str, job_config_data['exception_type'])

        if self.job_details.instance is None:
            raise RuntimeError('Instance is not initialized')

        dataset_type = self.job_details.instance.get('data_source', 'swebench')
        func = get_registered_functions(function_type, dataset_type)

        try:
            if func is None:
                raise FunctionNotRegisteredError(
                    f"Function '{dataset_type}' not found in registry type '{function_type}'"
                )

            # Enter appropriate phase - all work here counts toward timeout
            if self.job_details.timer is None:
                raise RuntimeError('Timer is not initialized')

            with phase_context(self.job_details.timer, function_type):
                # Execute the appropriate function based on job type
                if job_type == JobType.INIT:
                    # Use timeout-aware coroutine execution
                    init_coro = func(
                        job_details=self.job_details,
                        sid=self.job_id,
                    )
                    runtime, metadata, config = await run_with_timeout_awareness(
                        self.job_details.timer, init_coro, self.job_details
                    )
                    self.job_details.runtime = runtime
                    self.job_details.metadata = metadata
                    self.job_details.config = config

                elif job_type == JobType.RUN:
                    # Use timeout-aware coroutine execution
                    run_coro = func(
                        job_details=self.job_details,
                        sid=self.job_id,
                    )
                    run_results = await run_with_timeout_awareness(
                        self.job_details.timer, run_coro, self.job_details
                    )
                    self.job_details.run_results = run_results
                    # Close runtime (automatically in "others" phase - doesn't count toward timeout)
                    if self.job_details.runtime:
                        self._cleanup_job_runtime(self.job_details.runtime, self.job_id)
                        self.job_details.runtime = None

                elif job_type == JobType.EVAL:
                    # Use timeout-aware coroutine execution
                    eval_coro = func(
                        job_details=self.job_details,
                        sid=f'eval_{self.job_id}',
                        allow_skip=self.allow_skip_eval,
                        reward=self.reward,
                    )
                    eval_report = await run_with_timeout_awareness(
                        self.job_details.timer, eval_coro, self.job_details
                    )
                    # Only keep the 'report' field if present
                    if isinstance(eval_report, dict) and 'report' in eval_report:
                        self.job_details.eval_results = eval_report['report']
                    else:
                        self.job_details.eval_results = eval_report
                    if self.job_details.event is not None:
                        self.job_details.event.set()

        except TimeoutError as e:
            self.job_details.timeout_error = True

            # Handle runtime cleanup for run workers
            if job_type == JobType.RUN and self.job_details.runtime:
                self._cleanup_job_runtime(self.job_details.runtime, self.job_id)
                self.job_details.runtime = None

            exception_func = get_registered_functions(exception_type, dataset_type)
            if exception_func is not None:
                self.job_details.results = exception_func(self.job_details, e)
            else:
                self.job_details.results = {
                    'error': f'Timeout during {function_type}: {str(e)}',
                    'timeout': True,
                }
            if self.job_details.event is not None:
                self.job_details.event.set()

        except Exception as e:
            # Handle runtime cleanup for run workers
            if job_type == JobType.RUN and self.job_details.runtime:
                self._cleanup_job_runtime(self.job_details.runtime, self.job_id)
                self.job_details.runtime = None

            exception_func = get_registered_functions(exception_type, dataset_type)
            if exception_func is not None:
                self.job_details.results = exception_func(self.job_details, e)
            else:
                self.job_details.results = {
                    'error': f'Exception during {function_type}: {str(e)}'
                }
            if self.job_details.event is not None:
                self.job_details.event.set()

    def run(self):
        # Run initialization step
        if self.concurrency_control is not None:
            self.concurrency_control.init.acquire()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status[self.job_id] = JobType.INIT
        self.initialize()
        asyncio.run(self.run_step(JobType.INIT))
        if self.concurrency_control is not None:
            self.concurrency_control.init.release()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status.pop(self.job_id)

        if self.job_details.event is not None and self.job_details.event.is_set():
            return self.get_results()

        # Run execution step
        if self.concurrency_control is not None:
            self.concurrency_control.run.acquire()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status[self.job_id] = JobType.RUN
        asyncio.run(self.run_step(JobType.RUN))
        if self.concurrency_control is not None:
            self.concurrency_control.run.release()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status.pop(self.job_id)
        if self.job_details.event is not None and self.job_details.event.is_set():
            return self.get_results()

        # Run evaluation step
        if self.concurrency_control is not None:
            self.concurrency_control.eval.acquire()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status[self.job_id] = JobType.EVAL
        asyncio.run(self.run_step(JobType.EVAL))
        if self.concurrency_control is not None:
            self.concurrency_control.eval.release()
        if self.job_status is not None:
            with self.job_status_lock:
                self.job_status.pop(self.job_id)
        return self.get_results()

    def get_results(self):
        dataset_type = self.job_details.instance.get('data_source', 'swebench')
        # Get final result
        _final_result_func = get_registered_functions('final_result', dataset_type)
        if _final_result_func is None:
            result: dict[str, Any] = {
                'critical_error': 'final_result',
                'error': f'Function not found in registry type final_result for dataset type {dataset_type}',
            }
        else:
            result = _final_result_func(self.job_details)

        # Add timing information to result
        if self.job_details.timer:
            timing_info = self.job_details.timer.get_timing_info()
            result['timing'] = timing_info
        return result


def process_job(
    job_id: str,
    instance: dict,
    sampling_params: dict,
    timeout: float,
    llm_server_addresses: str,
    allow_skip_eval: bool = True,
    reward_server_ip: Union[list[str], None] = None,
    result_queue: Union[mp.Queue, None] = None,
    concurrency_control: Union[ConcurrencyControl, None] = None,
    job_status: Union[dict, None] = None,
    job_status_lock: Union[mp.Lock, None] = None,
):
    try:
        os.setsid()
    except Exception:
        pass

    pgid = os.getpgid(0)
    print(f'[Worker {job_id}] PID={os.getpid()} PGID={pgid}')
    worker = Worker(
        job_id,
        instance,
        sampling_params,
        timeout,
        llm_server_addresses,
        allow_skip_eval,
        reward_server_ip,
        concurrency_control,
        job_status,
        job_status_lock,
    )
    try:
        result = worker.run()
    except Exception as e:
        result = {'error': str(e)}
    result_queue.put({'job_id': job_id, 'result': result})


class JobState:
    def __init__(
        self,
        job_id: str,
        finished: threading.Event,
        process: mp.Process,
    ):
        self.job_id = job_id
        self.finished = finished
        self.process = process
        self.result = None

    def close(self):
        """Clean up job state resources"""
        try:
            self.finished.set()
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5.0)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1.0)
        except Exception:
            # Ignore errors during cleanup
            pass

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except Exception:
            pass


class OpenHandsServer:
    def __init__(
        self,
        llm_server_addresses: list[str] | None = None,
        max_init_workers: int = 6,
        max_run_workers: int = 5,
        max_eval_workers: int | None = None,
        allow_skip_eval: bool = True,
        reward_server_ip: list[str] | None = None,
    ):
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        # If eval workers not specified, mirror run_workers
        self.max_eval_workers = (
            max_run_workers if max_eval_workers is None else max_eval_workers
        )

        if llm_server_addresses is None:
            llm_server_addresses = []
        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)
        self._address_lock = threading.RLock()

        self.allow_skip_eval = allow_skip_eval
        self.reward_server_ip = reward_server_ip

        self._job_details_lock = threading.RLock()
        self.jobs: dict[str, JobState] = {}

        self.running = False
        self.result_queue = mp.Queue()

        self.concurrency_control = None
        self.manager = None
        self.job_status = None
        self.job_status_lock = None

        self._result_check_thread = None

    def _start_result_check_thread(self):
        self._result_check_thread = threading.Thread(
            target=self._result_check_thread_func
        )
        self._result_check_thread.start()

    def _result_check_thread_func(self):
        while True:
            results = self.result_queue.get()
            if results is None:
                break
            job_id = results['job_id']
            result = results['result']
            with self._job_details_lock:
                if job_id in self.jobs:
                    self.jobs[job_id].result = result
                    self.jobs[job_id].finished.set()

    def _stop_result_check_thread(self):
        self.result_queue.put(None)
        self._result_check_thread.join()
        self._result_check_thread = None

    def get_unique_id(self, instance, max_retries=10):
        base = f'{get_instance_id(instance)}_{instance["trajectory_id"]}'
        base_hash = hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]
        for _ in range(max_retries):
            rand = uuid.uuid4().hex[:8]
            uid = f'{base_hash}_{rand}'
            with self._job_details_lock:
                if uid not in self.jobs:
                    return uid
        raise ValueError('Failed to get unique id')

    def add_llm_server_address(self, llm_server_address: str):
        with self._address_lock:
            # Check if address already exists
            for weight, addr in self.weighted_addresses:
                if addr == llm_server_address:
                    logger.warning(
                        f'Warning: LLM server address {llm_server_address} already exists'
                    )
                    return

            heapq.heappush(self.weighted_addresses, [0, llm_server_address])
            logger.info(f'Added LLM server address: {llm_server_address}')

    def clear_llm_server_addresses(self):
        with self._address_lock:
            self.weighted_addresses.clear()
            logger.info('Cleared LLM server addresses')

    def get_llm_server_addresses(self):
        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1  # type: ignore
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
            return address

    def cancel_job(self, job_id: str):
        if not self.running:
            raise RuntimeError('Server is not running')
        if job_id not in self.jobs:
            raise ValueError(f'Job {job_id} not found')
        with self._job_details_lock:
            job_state = self.jobs.get(job_id)
            if job_state is not None:
                job_state.close()
                del self.jobs[job_id]
        if job_id in self.job_status:
            with self.job_status_lock:
                status = self.job_status.pop(job_id)
                match status:
                    case JobType.INIT:
                        self.concurrency_control.init.release()
                    case JobType.RUN:
                        self.concurrency_control.run.release()
                    case JobType.EVAL:
                        self.concurrency_control.eval.release()

    def process(
        self,
        instance,
        sampling_params,
        job_id: str | None = None,
        timeout: float = 300.0,
    ):
        """Process a single request by launching a process_job"""
        if not self.running:
            raise RuntimeError('Server is not running')

        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

        dataset_type = instance.get('data_source', 'swebench')
        if not is_registered_handler(dataset_type):
            raise FunctionNotRegisteredError(
                f'Dataset type {dataset_type} is not registered'
            )

        if job_id is None:
            job_id = self.get_unique_id(instance)

        llm_server_addresses = self.get_llm_server_addresses()

        args = (
            job_id,
            instance,
            sampling_params,
            timeout,
            llm_server_addresses,
            self.allow_skip_eval,
            self.reward_server_ip,
            self.result_queue,
            self.concurrency_control,
            self.job_status,
            self.job_status_lock,
        )
        # Start the process
        p = mp.Process(target=process_job, args=args)
        logger.info(f'Starting process {job_id}.')
        p.start()

        finished = threading.Event()

        # Create job state and add to tracking
        job_state = JobState(job_id, finished, p)
        self.jobs[job_id] = job_state

        # Get result from queue
        self.jobs[job_id].finished.wait()
        result = dict(self.jobs[job_id].result)

        with self._job_details_lock:
            self.jobs[job_id].close()
            del self.jobs[job_id]
        logger.info(f'Job {job_id} deleted from jobs')
        logger.info(f'Finished process {job_id}')
        return result

    def clear_singularity_jobs(self):
        kill_all_singularity_jobs(self._exclude_pids)

    def status(self):
        if self.job_status is None:
            return {
                'running': self.running,
                'jobs': len(self.jobs),
            }
        with self.job_status_lock:
            init_jobs = [
                job_id
                for job_id, job_type in self.job_status.items()
                if job_type == JobType.INIT
            ]
            run_jobs = [
                job_id
                for job_id, job_type in self.job_status.items()
                if job_type == JobType.RUN
            ]
            eval_jobs = [
                job_id
                for job_id, job_type in self.job_status.items()
                if job_type == JobType.EVAL
            ]
        return {
            'running': self.running,
            'jobs': len(self.jobs),
            'init': len(init_jobs),
            'run': len(run_jobs),
            'eval': len(eval_jobs),
        }

    def start(self):
        """Start the server and all workers"""
        if self.running:
            return

        self.running = True
        logger.info('Starting server...')

        with self._address_lock:
            self._exclude_pids = get_singularity_job_pids()
            logger.info(f'Excluded Singularity job PIDs: {self._exclude_pids}')

        self._start_result_check_thread()

        self.clear_singularity_jobs()

        if self.concurrency_control is None:
            self.concurrency_control = ConcurrencyControl(
                max_init_workers=self.max_init_workers,
                max_run_workers=self.max_run_workers,
                max_eval_workers=self.max_eval_workers,
            )

        if self.manager is None:
            self.manager = mp.Manager()
            self.job_status = self.manager.dict()
            self.job_status_lock = self.manager.Lock()

    def stop(self):
        """Stop the server and wait for all jobs to complete"""
        if not self.running:
            return

        logger.info('Stopping server...')
        self.running = False

        # Wait for all remaining jobs to complete
        remaining_jobs = list(self.jobs.keys())
        for job_id in remaining_jobs:
            with self._job_details_lock:
                job_state = self.jobs.get(job_id)
                if job_state is not None:
                    job_state.close()
                    del self.jobs[job_id]

        with self._job_details_lock:
            self.jobs.clear()

        self.clear_singularity_jobs()

        if self.concurrency_control is not None:
            self.concurrency_control.release_all()

        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
            self.job_status = None
            self.job_status_lock = None

        clear_queue(self.result_queue)
        self._stop_result_check_thread()
        logger.info('Server stopped')
