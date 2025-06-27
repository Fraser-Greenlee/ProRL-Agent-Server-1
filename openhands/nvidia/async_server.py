import asyncio
import heapq
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia import register_swe_agent_functions
from openhands.nvidia.registry import (
    FunctionNotRegisteredError,
    JobDetails,
    get_registered_functions,
    is_registered_handler,
)
from openhands.nvidia.utils import clear_queue

register_swe_agent_functions()


class OpenHandsServer:
    def __init__(
        self,
        llm_server_addresses: list[str] | None = None,
        max_init_workers: int = 6,
        max_run_workers: int = 5,
        max_eval_workers: int | None = None,
        allow_skip_eval: bool = True,
    ):
        """Create server.

        If *max_eval_workers* is not provided, it defaults to the same value as
        *max_run_workers*, so you only need to specify one number when you want
        these two pools to have the same size.

        allow_skip_eval: if True, skip evaluation if git_patch is None or empty.
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

        self._server_running: bool = False

        # store job detail objects to pass around.
        self._job_details: dict[str, JobDetails] = {}

        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)

    def get_unique_id(self, instance, max_retries=10):
        for _ in range(max_retries):
            uid = str(uuid.uuid4())
            uid = f'{instance.instance_id}_{instance.trajectory_id}_{uid}'
            if uid not in self._job_details:
                return uid
        raise ValueError('Failed to get unique id')

    def add_llm_server_address(self, llm_server_address: str):
        heapq.heappush(self.weighted_addresses, [0, llm_server_address])

    def create_llm_config(self, sampling_params):
        if len(self.weighted_addresses) == 0:
            raise ValueError('No LLM server addresses added')

        address = self.weighted_addresses[0][1]
        self.weighted_addresses[0][0] += 1  # type: ignore
        heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        llm_config = LLMConfig(base_url=address, **sampling_params)
        return llm_config

    def start(self):
        if self._server_running:
            raise RuntimeError('Server is already running')
        self._server_running = True

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

    def process(self, instance, sampling_params, job_id=None):
        if not self._server_running:
            raise RuntimeError('Server is not running')

        if len(self.weighted_addresses) == 0:
            raise ValueError('No LLM server addresses added')

        dataset_type = getattr(instance, 'data_source', 'swebench')
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
        job_details.start_time = time.time()
        job_details.event = threading.Event()
        self._job_details[job_id] = job_details
        print(f'Job {job_id} added to job details')

        # Add job to init queue
        self.init_queue.put(job_id)
        print(f'Job {job_id} added to init queue')

        # Wait for job to be finished
        job_details.event.wait()
        job_details.end_time = time.time()

        # Get final result
        _final_result_func = get_registered_functions('final_result', dataset_type)
        if _final_result_func is None:
            result = {
                'critical_error': 'final_result',
                'error': f'Function not found in registry type final_result for dataset type {dataset_type}',
            }
        else:
            result = _final_result_func(job_details)

        # Close runtime
        if job_details.runtime:
            job_details.runtime.close()
        # Delete job details
        del self._job_details[job_id]
        return result

    async def _init_worker(self, wid):
        while True:
            print(f'[init-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.init_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                print(f'[init-worker-{wid}] Received stop signal, exiting')
                self.init_queue.task_done()
                break

            print(f'[init-worker-{wid}] Got job {job_id}')
            job_details = self._job_details[job_id]
            self._active_init_jobs.add(job_id)
            dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
            _init_func = get_registered_functions('init', dataset_type)
            try:
                if _init_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'init'"
                    )
                runtime, metadata, config = await _init_func(
                    job_details.instance,
                    job_details.llm_config,
                    sid=job_id,
                    max_iterations=job_details.max_iterations,
                )
                job_details.runtime = runtime
                job_details.metadata = metadata
                job_details.config = config
                self.run_queue.put(job_id)
            except Exception as e:
                _init_exception_func = get_registered_functions(
                    'init_exception', dataset_type
                )
                job_details.results = _init_exception_func(job_details, e)
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                self._active_init_jobs.remove(job_id)
                self.init_queue.task_done()

    async def _run_worker(self, wid):
        while True:
            print(f'[run-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.run_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                print(f'[run-worker-{wid}] Received stop signal, exiting')
                self.run_queue.task_done()
                break

            job_details = self._job_details[job_id]
            print(f'[run-worker-{wid}] Got job {job_id}')
            job_details.start_run_time = time.time()
            self._active_run_jobs.add(job_id)
            dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
            _run_func = get_registered_functions('run', dataset_type)
            try:
                if _run_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'run'"
                    )
                run_results = await _run_func(
                    job_details.runtime,
                    job_details.metadata,
                    job_details.config,
                    job_details.instance,
                )
                job_details.run_results = run_results

                # Close runtime right after run finishes (before evaluation)
                if job_details.runtime:
                    job_details.runtime.close()
                    job_details.runtime = None

                # Push to evaluation queue for further processing
                self.evaluate_queue.put(job_id)
            except Exception as e:
                # Ensure runtime is closed even if an exception occurs
                if job_details.runtime:
                    job_details.runtime.close()
                    job_details.runtime = None

                _run_exception_func = get_registered_functions(
                    'run_exception', dataset_type
                )
                job_details.results = _run_exception_func(job_details, e)
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                self._active_run_jobs.remove(job_id)
                self.run_queue.task_done()

    async def _eval_worker(self, wid):
        """Worker that evaluates the generated patch and produces a report."""
        while True:
            print(f'[eval-worker-{wid}] Waiting for job')
            job_id = await asyncio.to_thread(self.evaluate_queue.get)

            # Check for stop sentinel
            if job_id == '__STOP__':
                print(f'[eval-worker-{wid}] Received stop signal, exiting')
                self.evaluate_queue.task_done()
                break

            print(f'[eval-worker-{wid}] Got job {job_id}')
            job_details = self._job_details[job_id]
            job_details.start_eval_time = time.time()
            self._active_eval_jobs.add(job_id)
            dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
            _eval_func = get_registered_functions('eval', dataset_type)
            try:
                if _eval_func is None:
                    raise FunctionNotRegisteredError(
                        f"Function '{dataset_type}' not found in registry type 'eval'"
                    )
                eval_report = await _eval_func(
                    job_details,
                    sid=f'eval_{job_id}',
                    allow_skip=self.allow_skip_eval,
                )
                # Only keep the 'report' field if present
                if isinstance(eval_report, dict) and 'report' in eval_report:
                    job_details.eval_results = eval_report['report']
                else:
                    job_details.eval_results = eval_report
                if job_details.event is not None:
                    job_details.event.set()
            except Exception as e:
                _eval_exception_func = get_registered_functions(
                    'eval_exception', dataset_type
                )
                job_details.results = _eval_exception_func(job_details, e)
                if job_details.event is not None:
                    job_details.event.set()
            finally:
                self._active_eval_jobs.remove(job_id)
                self.evaluate_queue.task_done()

    def _run_worker_in_thread(self, worker_id, is_init_worker):
        """Run a worker in its own thread with its own event loop. Run until the worker is stopped."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if is_init_worker:
            loop.run_until_complete(self._init_worker(worker_id))
            self._init_workers[worker_id] = loop
        else:
            loop.run_until_complete(self._run_worker(worker_id))
            self._run_workers[worker_id] = loop

    def _run_eval_worker_in_thread(self, worker_id):
        """Start an evaluation worker in its own thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._eval_worker(worker_id))
        self._eval_workers[worker_id] = loop

    def stop(self):
        """Stops the server by shutting down all workers and clearing all queues and jobs."""
        if not self._server_running:
            return
        print('Stopping events')
        # Signal all active jobs to complete
        for job_id in list(self._active_init_jobs):
            job = self._job_details.get(job_id)
            if job is not None and job.event is not None:
                job.event.set()

        for job_id in list(self._active_run_jobs):
            job = self._job_details.get(job_id)
            if job is not None and job.event is not None:
                job.event.set()

        for job_id in list(self._active_eval_jobs):
            job = self._job_details.get(job_id)
            if job is not None and job.event is not None:
                job.event.set()

        print('Stopping queues')
        # Add sentinel values to queues to unblock workers
        for _ in range(self.max_init_workers):
            try:
                self.init_queue.put_nowait('__STOP__')
            except Exception as e:
                print(f'Warning: Failed to put stop signal in init queue: {e}')

        for _ in range(self.max_run_workers):
            try:
                self.run_queue.put_nowait('__STOP__')
            except Exception as e:
                print(f'Warning: Failed to put stop signal in run queue: {e}')

        for _ in range(self.max_eval_workers):
            try:
                self.evaluate_queue.put_nowait('__STOP__')
            except Exception as e:
                print(f'Warning: Failed to put stop signal in eval queue: {e}')

        print('Shutting down executor')
        # Shutdown the executor with a timeout - this is the main fix
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

        print('Clearing active jobs')
        # Clear all state
        self._active_init_jobs.clear()
        self._active_run_jobs.clear()
        self._active_eval_jobs.clear()
        self._job_details.clear()
        self._init_workers.clear()
        self._run_workers.clear()
        self._eval_workers.clear()
        # Reset queues
        clear_queue(self.init_queue)
        clear_queue(self.run_queue)
        clear_queue(self.evaluate_queue)

        self._server_running = False
        print(f'Server status: {self.status()}')

    def status(self):
        """Returns the number of jobs currently being processed in both queues and workers."""
        init_queue_count = self.init_queue.qsize()
        run_queue_count = self.run_queue.qsize()
        eval_queue_count = self.evaluate_queue.qsize()

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
