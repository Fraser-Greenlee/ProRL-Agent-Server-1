import asyncio
import heapq
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

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
        """Create server.

        If *max_eval_workers* is not provided, it defaults to the same value as
        *max_run_workers*, so you only need to specify one number when you want
        these two pools to have the same size.

        allow_skip_eval: if True, skip evaluation if run_results (i.e. git_patch) is None or empty.
        Set to False for testing.
        """
        if llm_server_addresses is None:
            llm_server_addresses = []
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.allow_skip_eval = allow_skip_eval
        # If eval workers not specified, mirror run_workers
        self.max_eval_workers = (
            max_run_workers if max_eval_workers is None else max_eval_workers
        )

        self.init_queue: queue.Queue[str] = queue.Queue()
        self.run_queue: queue.Queue[str] = queue.Queue()
        self.evaluate_queue: queue.Queue[str] = queue.Queue()
        self._init_workers: list[asyncio.AbstractEventLoop | None] = []
        self._run_workers: list[asyncio.AbstractEventLoop | None] = []
        self._eval_workers: list[asyncio.AbstractEventLoop | None] = []
        self._active_init_jobs: set[str] = set()  # Track jobs being initialized
        self._active_run_jobs: set[str] = set()  # Track jobs being run
        self._active_eval_jobs: set[str] = set()  # Track jobs being evaluated
        self._exclude_pids: set[str] = set()

        self._server_running: bool = False

        # store job detail objects to pass around.
        self._job_details: dict[str, JobDetails] = {}

        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)

        # THREAD SAFETY: Add locks to protect shared data structures
        self._state_lock = threading.RLock()  # Reentrant lock for active job sets
        self._job_details_lock = threading.RLock()  # Separate lock for job details dict
        self._address_lock = threading.RLock()  # Separate lock for address list

        self.reward: Optional[Reward] = None
        if reward_server_ip is not None:
            logger.info(f'Setting up reward with server IP: {reward_server_ip}')
            self.reward = Reward(server_ip=reward_server_ip)
        else:
            logger.warning(
                'No reward server IP provided. Evaluations would only work for swebench problems.'
            )

    def get_unique_id(self, instance, max_retries=10):
        for _ in range(max_retries):
            uid = str(uuid.uuid4())
            uid = f'{get_instance_id(instance)}_{instance["trajectory_id"]}_{uid}'
            with self._job_details_lock:
                if uid not in self._job_details:
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

    def clear_singularity_jobs(self):
        kill_all_singularity_jobs(self._exclude_pids)

    def create_llm_config(self, sampling_params):
        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1  # type: ignore
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        llm_config = LLMConfig(base_url=address, **sampling_params)
        return llm_config

    def _cleanup_job_runtime(self, runtime, job_id: str):
        """Comprehensive cleanup of runtime resources to prevent thread leakage."""
        try:
            # 1. Close event stream and its thread pools
            if hasattr(runtime, 'event_stream') and runtime.event_stream:
                try:
                    runtime.event_stream.close()
                    logger.debug(f'Event stream closed for job {job_id}')
                except Exception as e:
                    logger.warning(f'Error closing event stream for job {job_id}: {e}')

            # 2. Close runtime (handles container processes, plugins, etc.)
            runtime.close()

            # 3. Force cleanup any remaining subprocess-related resources
            time.sleep(0.1)  # Brief pause for cleanup to complete

        except Exception as e:
            logger.error(
                f'Error in comprehensive runtime cleanup for job {job_id}: {e}'
            )
            # Don't re-raise - we want cleanup to continue even if parts fail

    def start(self):
        if self._server_running:
            raise RuntimeError('Server is already running')
        self._server_running = True

        with self._address_lock:
            self._exclude_pids = get_singularity_job_pids()
            logger.info(f'Excluded Singularity job PIDs: {self._exclude_pids}')

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_init_workers
            + self.max_run_workers
            + self.max_eval_workers
        )

        # Initialize worker lists
        self._init_workers = [None] * self.max_init_workers
        self._run_workers = [None] * self.max_run_workers
        self._eval_workers = [None] * self.max_eval_workers

        # Submit init workers
        for i in range(self.max_init_workers):
            self._executor.submit(self._run_worker_in_thread, i, True)

        # Submit run workers
        for i in range(self.max_run_workers):
            self._executor.submit(self._run_worker_in_thread, i, False)

        # Submit evaluation workers
        for i in range(self.max_eval_workers):
            self._executor.submit(self._run_eval_worker_in_thread, i)

        self.clear_singularity_jobs()

    def process(self, instance, sampling_params, job_id=None, timeout: float = 300.0):
        if not self._server_running:
            raise RuntimeError('Server is not running')

        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

        dataset_type = instance.get('data_source', 'swebench')
        if not is_registered_handler(dataset_type):
            raise FunctionNotRegisteredError(
                f'Dataset type {dataset_type} is not registered'
            )

        # Create job details
        if job_id is None:
            job_id = self.get_unique_id(instance)
        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = instance
        if 'max_iterations' in sampling_params:
            job_details.max_iterations = sampling_params.pop('max_iterations')
        llm_config = self.create_llm_config(sampling_params)
        job_details.llm_config = llm_config
        job_details.event = threading.Event()

        # Initialize timer - only tracks init/run/eval phases
        # All other time is automatically counted as "others" (not counted toward timeout)
        job_details.timer = PausableTimer(timeout=timeout)
        job_details.timer.start()

        with self._job_details_lock:
            self._job_details[job_id] = job_details
        logger.info(f'Job {job_id} added to job details')

        # Add job to init queue
        self.init_queue.put(job_id)
        logger.info(f'Job {job_id} added to init queue')

        # Wait for job to be finished
        job_details.event.wait()

        # Get final result
        _final_result_func = get_registered_functions('final_result', dataset_type)
        if _final_result_func is None:
            result: dict[str, Any] = {
                'critical_error': 'final_result',
                'error': f'Function not found in registry type final_result for dataset type {dataset_type}',
            }
        else:
            result = _final_result_func(job_details)

        # Add timing information to result
        if job_details.timer:
            timing_info = job_details.timer.get_timing_info()
            result['timing'] = timing_info

        # Close runtime
        if job_details.runtime:
            self._cleanup_job_runtime(job_details.runtime, job_id)
        # Delete job details
        with self._job_details_lock:
            del self._job_details[job_id]
        return result

    async def _init_worker(self, wid):
        while True:
            logger.info(f'[init-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.init_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                logger.info(f'[init-worker-{wid}] Received stop signal, exiting')
                self.init_queue.task_done()
                break

            logger.info(f'[init-worker-{wid}] Got job {job_id}')

            # Thread-safe job details retrieval
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(
                        f'[init-worker-{wid}] Job {job_id} not found, skipping'
                    )
                    self.init_queue.task_done()
                    continue

            # Thread-safe active jobs tracking
            with self._state_lock:
                self._active_init_jobs.add(job_id)

            if job_details.instance is None:
                raise RuntimeError('Instance is not initialized')
            dataset_type = job_details.instance.get('data_source', 'swebench')
            _init_func = get_registered_functions('init', dataset_type)

            try:
                if _init_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'init'"
                    )

                # Enter init phase - all work here counts toward timeout
                if job_details.timer is None:
                    raise RuntimeError('Timer is not initialized')
                with phase_context(job_details.timer, 'init'):
                    # Use timeout-aware coroutine execution
                    init_coro = _init_func(
                        job_details=job_details,
                        sid=job_id,
                        max_iterations=job_details.max_iterations,
                    )
                    runtime, metadata, config = await run_with_timeout_awareness(
                        job_details.timer, init_coro
                    )

                job_details.runtime = runtime
                job_details.metadata = metadata
                job_details.config = config

                # Put in run queue (automatically becomes "others" phase)
                self.run_queue.put(job_id)

            except TimeoutError as e:
                logger.warning(
                    f'[init-worker-{wid}] Job {job_id} timed out during init: {e}'
                )
                job_details.timeout_error = True
                _init_exception_func = get_registered_functions(
                    'init_exception', dataset_type
                )
                if _init_exception_func is not None:
                    job_details.results = _init_exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Timeout during init: {str(e)}',
                        'timeout': True,
                    }
                if job_details.event is not None:
                    job_details.event.set()
            except Exception as e:
                _init_exception_func = get_registered_functions(
                    'init_exception', dataset_type
                )
                if _init_exception_func is not None:
                    job_details.results = _init_exception_func(job_details, e)
                else:
                    job_details.results = {'error': f'Exception during init: {str(e)}'}
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                # Thread-safe cleanup
                with self._state_lock:
                    self._active_init_jobs.discard(job_id)
                self.init_queue.task_done()

    async def _run_worker(self, wid):
        while True:
            logger.info(f'[run-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.run_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                logger.info(f'[run-worker-{wid}] Received stop signal, exiting')
                self.run_queue.task_done()
                break

            # Thread-safe job details retrieval
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(
                        f'[run-worker-{wid}] Job {job_id} not found, skipping'
                    )
                    self.run_queue.task_done()
                    continue

            logger.info(f'[run-worker-{wid}] Got job {job_id}')

            # Thread-safe active jobs tracking
            with self._state_lock:
                self._active_run_jobs.add(job_id)

            if job_details.instance is None:
                raise RuntimeError('Instance is not initialized')
            dataset_type = job_details.instance.get('data_source', 'swebench')
            _run_func = get_registered_functions('run', dataset_type)

            try:
                if _run_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'run'"
                    )

                # Enter run phase - all work here counts toward timeout
                if job_details.timer is None:
                    raise RuntimeError('Timer is not initialized')
                with phase_context(job_details.timer, 'run'):
                    # Use timeout-aware coroutine execution
                    run_coro = _run_func(
                        job_details=job_details,
                        sid=job_id,
                    )
                    run_results = await run_with_timeout_awareness(
                        job_details.timer, run_coro
                    )

                job_details.run_results = run_results

                # Close runtime (automatically in "others" phase - doesn't count toward timeout)
                if job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

                # Push to evaluation queue (automatically "others" phase)
                self.evaluate_queue.put(job_id)

            except TimeoutError as e:
                logger.warning(
                    f'[run-worker-{wid}] Job {job_id} timed out during run: {e}'
                )
                job_details.timeout_error = True
                # Ensure runtime is closed (automatically in "others" phase)
                if job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

                _run_exception_func = get_registered_functions(
                    'run_exception', dataset_type
                )
                if _run_exception_func is not None:
                    job_details.results = _run_exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Timeout during run: {str(e)}',
                        'timeout': True,
                    }
                if job_details.event is not None:
                    job_details.event.set()
            except Exception as e:
                # Ensure runtime is closed (automatically in "others" phase)
                if job_details.runtime:
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    job_details.runtime = None

                _run_exception_func = get_registered_functions(
                    'run_exception', dataset_type
                )
                if _run_exception_func is not None:
                    job_details.results = _run_exception_func(job_details, e)
                else:
                    job_details.results = {'error': f'Exception during run: {str(e)}'}
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                # Thread-safe cleanup
                with self._state_lock:
                    self._active_run_jobs.discard(job_id)
                self.run_queue.task_done()

    async def _eval_worker(self, wid):
        """Worker that evaluates the generated patch and produces a report."""
        while True:
            logger.info(f'[eval-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.evaluate_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                logger.info(f'[eval-worker-{wid}] Received stop signal, exiting')
                self.evaluate_queue.task_done()
                break

            logger.info(f'[eval-worker-{wid}] Got job {job_id}')

            # Thread-safe job details retrieval
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(
                        f'[eval-worker-{wid}] Job {job_id} not found, skipping'
                    )
                    self.evaluate_queue.task_done()
                    continue

            # Thread-safe active jobs tracking
            with self._state_lock:
                self._active_eval_jobs.add(job_id)

            if job_details.instance is None:
                raise RuntimeError('Instance is not initialized')
            dataset_type = job_details.instance.get('data_source', 'swebench')
            _eval_func = get_registered_functions('eval', dataset_type)

            try:
                if _eval_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'eval'"
                    )

                # Enter eval phase - all work here counts toward timeout
                if job_details.timer is None:
                    raise RuntimeError('Timer is not initialized')
                with phase_context(job_details.timer, 'eval'):
                    # Use timeout-aware coroutine execution
                    eval_coro = _eval_func(
                        job_details,
                        sid=f'eval_{job_id}',
                        allow_skip=self.allow_skip_eval,
                        reward=self.reward,
                    )
                    eval_report = await run_with_timeout_awareness(
                        job_details.timer, eval_coro
                    )

                # Only keep the 'report' field if present
                if isinstance(eval_report, dict) and 'report' in eval_report:
                    job_details.eval_results = eval_report['report']
                else:
                    job_details.eval_results = eval_report

                if job_details.event is not None:
                    job_details.event.set()

            except TimeoutError as e:
                logger.warning(
                    f'[eval-worker-{wid}] Job {job_id} timed out during eval: {e}'
                )
                job_details.timeout_error = True
                _eval_exception_func = get_registered_functions(
                    'eval_exception', dataset_type
                )
                if _eval_exception_func is not None:
                    job_details.results = _eval_exception_func(job_details, e)
                else:
                    job_details.results = {
                        'error': f'Timeout during eval: {str(e)}',
                        'timeout': True,
                    }
                if job_details.event is not None:
                    job_details.event.set()
            except Exception as e:
                _eval_exception_func = get_registered_functions(
                    'eval_exception', dataset_type
                )
                if _eval_exception_func is not None:
                    job_details.results = _eval_exception_func(job_details, e)
                else:
                    job_details.results = {'error': f'Exception during eval: {str(e)}'}
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                # Thread-safe cleanup
                with self._state_lock:
                    self._active_eval_jobs.discard(job_id)
                self.evaluate_queue.task_done()

    def _run_worker_in_thread(self, worker_id, is_init_worker):
        """Run a worker in its own thread with its own event loop. Run until the worker is stopped."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if is_init_worker:
                self._init_workers[worker_id] = loop
                loop.run_until_complete(self._init_worker(worker_id))
            else:
                self._run_workers[worker_id] = loop
                loop.run_until_complete(self._run_worker(worker_id))
        finally:
            # Always close the event loop to prevent resource leaks
            try:
                loop.close()
                logger.debug(
                    f'Event loop closed for {"init" if is_init_worker else "run"} worker {worker_id}'
                )
            except Exception as e:
                logger.warning(
                    f'Error closing event loop for {"init" if is_init_worker else "run"} worker {worker_id}: {e}'
                )

    def _run_eval_worker_in_thread(self, worker_id):
        """Start an evaluation worker in its own thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            self._eval_workers[worker_id] = loop
            loop.run_until_complete(self._eval_worker(worker_id))
        finally:
            # Always close the event loop to prevent resource leaks
            try:
                loop.close()
                logger.debug(f'Event loop closed for eval worker {worker_id}')
            except Exception as e:
                logger.warning(
                    f'Error closing event loop for eval worker {worker_id}: {e}'
                )

    def stop(self):
        """Stops the server by shutting down all workers and clearing all queues and jobs."""
        if not self._server_running:
            return

        logger.info('Stopping OpenHands server...')

        # Step 1: Set server as not running to prevent new jobs
        self._server_running = False

        # Step 2: Force complete all active jobs with timeout errors
        logger.info('Signaling active jobs to complete...')
        with self._state_lock:
            active_jobs = (
                list(self._active_init_jobs)
                + list(self._active_run_jobs)
                + list(self._active_eval_jobs)
            )

        # Signal all active jobs and mark them with timeout errors
        for job_id in active_jobs:
            try:
                with self._job_details_lock:
                    job = self._job_details.get(job_id)
                    if job is not None:
                        # Mark as timed out and signal completion
                        job.timeout_error = True
                        if job.event is not None:
                            job.event.set()
                        # Clean up runtime if it exists
                        if job.runtime:
                            self._cleanup_job_runtime(job.runtime, job_id)
                            job.runtime = None
            except Exception as e:
                logger.warning(f'Error signaling job {job_id}: {e}')

        time.sleep(1)
        # Step 3: Add stop signals to all queues to unblock workers
        logger.info('Sending stop signals to worker queues...')
        for _ in range(self.max_init_workers):
            try:
                self.init_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in init queue: {e}')

        for _ in range(self.max_run_workers):
            try:
                self.run_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in run queue: {e}')

        for _ in range(self.max_eval_workers):
            try:
                self.evaluate_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f'Failed to put stop signal in eval queue: {e}')
        time.sleep(1)
        # Step 4: Shutdown executor with timeout
        logger.info('Shutting down thread pool executor...')
        if hasattr(self, '_executor') and self._executor:
            try:
                # Give workers 30 seconds to finish gracefully
                self._executor.shutdown(wait=True, cancel_futures=True)
                logger.info('Thread pool executor shutdown completed')
            except Exception as e:
                logger.warning(f'Error during executor shutdown: {e}')

        # Step 5: Force cleanup any remaining runtime resources
        logger.info('Cleaning up remaining job resources...')
        with self._job_details_lock:
            remaining_jobs = list(self._job_details.keys())
            for job_id in remaining_jobs:
                try:
                    job = self._job_details[job_id]
                    if job.runtime:
                        self._cleanup_job_runtime(job.runtime, job_id)
                        job.runtime = None
                except Exception as e:
                    logger.warning(f'Error cleaning up job {job_id}: {e}')

        # Step 6: Clear all data structures
        logger.info('Clearing internal data structures...')
        try:
            with self._state_lock:
                self._active_init_jobs.clear()
                self._active_run_jobs.clear()
                self._active_eval_jobs.clear()

            with self._job_details_lock:
                self._job_details.clear()

            # Clear worker lists (event loops are handled by thread completion)
            self._init_workers.clear()
            self._run_workers.clear()
            self._eval_workers.clear()
        except Exception as e:
            logger.warning(f'Error clearing data structures: {e}')

        # Step 7: Clear all queues
        logger.info('Clearing all queues...')
        try:
            clear_queue(self.init_queue)
            clear_queue(self.run_queue)
            clear_queue(self.evaluate_queue)
        except Exception as e:
            logger.warning(f'Error clearing queues: {e}')

        # Step 8: Clean up any remaining singularity jobs
        logger.info('Cleaning up singularity processes...')
        try:
            self.clear_singularity_jobs()
        except Exception as e:
            logger.warning(f'Error cleaning up singularity jobs: {e}')

        logger.info(f'Server stopped. Final status: {self.status()}')

    def status(self):
        """Returns the number of jobs currently being processed in both queues and workers."""
        init_queue_count = self.init_queue.qsize()
        run_queue_count = self.run_queue.qsize()
        eval_queue_count = self.evaluate_queue.qsize()

        # Thread-safe status reading
        with self._state_lock:
            active_init_count = len(self._active_init_jobs)
            active_run_count = len(self._active_run_jobs)
            active_eval_count = len(self._active_eval_jobs)

        return {
            'init_queue': init_queue_count,
            'run_queue': run_queue_count,
            'eval_queue': eval_queue_count,
            'active_init': active_init_count,
            'active_run': active_run_count,
            'active_eval': active_eval_count,
            'total': init_queue_count
            + run_queue_count
            + eval_queue_count
            + active_init_count
            + active_run_count
            + active_eval_count,
        }
