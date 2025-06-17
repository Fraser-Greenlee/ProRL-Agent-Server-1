import asyncio
from openhands.nvidia.swe_agent.utils import initialize_agents, run_agent
from openhands.core.config.llm_config import LLMConfig
from typing import List
import heapq
import uuid
import pandas as pd
from openhands.runtime.base import Runtime
from evaluation.utils.shared import EvalMetadata
from openhands.core.config import OpenHandsConfig
from concurrent.futures import ThreadPoolExecutor
import copy
import queue
import threading
import time

class JobDetails:
    job_id: str = None
    instance: pd.Series = None
    max_iterations: int = 2
    llm_config: LLMConfig = None
    runtime: Runtime = None
    metadata: EvalMetadata = None
    config: OpenHandsConfig = None
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event = None
    start_time: float = None
    start_run_time: float = None
    start_eval_time: float = None
    end_time: float = None

class OpenHandsServer:
    def __init__(
            self, 
            llm_server_addresses: List[str] = [],
            max_init_workers:int = 6,
            max_run_workers:int = 5,
            max_eval_workers:int | None = None,
            allow_skip_eval: bool = True,
        ):
        """Create server.

        If *max_eval_workers* is not provided, it defaults to the same value as
        *max_run_workers*, so you only need to specify one number when you want
        these two pools to have the same size.

        allow_skip_eval: if True, skip evaluation if git_patch is None or empty. 
        Set to False for testing.
        """
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.allow_skip_eval = allow_skip_eval
        # If eval workers not specified, mirror run_workers
        self.max_eval_workers = max_run_workers if max_eval_workers is None else max_eval_workers

        self.init_queue = None
        self.run_queue = None
        self.evaluate_queue = None
        self._init_workers = []
        self._run_workers = []
        self._active_init_jobs = set()  # Track jobs being initialized
        self._active_run_jobs = set()   # Track jobs being run
        self._active_eval_jobs = set()  # Track jobs being evaluated

        # store job detail objects to pass around.
        self._job_details = {}

        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)

    def get_unique_id(self, instance, max_retries=10):
        for _ in range(max_retries):
            uid = str(uuid.uuid4())
            uid = f"swebench_{instance.instance_id}_{instance.trajectory_id}_{uid}"
            if uid not in self._job_details:
                return uid
        raise ValueError("Failed to get unique id")

    def add_llm_server_address(self, llm_server_address: str):
        heapq.heappush(self.weighted_addresses, [0, llm_server_address])

    def create_llm_config(self, sampling_params):
        if len(self.weighted_addresses) == 0:
            raise ValueError("No LLM server addresses added")

        address = self.weighted_addresses[0][1]
        self.weighted_addresses[0][0] += 1
        heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        llm_config = LLMConfig(
            base_url = address,
            **sampling_params
        )
        return llm_config

    def start(self):
        self.init_queue = queue.Queue()
        self.run_queue = queue.Queue()
        self.evaluate_queue = queue.Queue()

        self._executor = ThreadPoolExecutor(max_workers=self.max_init_workers + self.max_run_workers + self.max_eval_workers)

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
        if len(self.weighted_addresses) == 0:
            raise ValueError("No LLM server addresses added")

        if not hasattr(self, 'init_queue') or self.init_queue is None:
            raise RuntimeError("Server is not started or has been stopped")

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
        print(f"Job {job_id} added to job details")

        # Add job to init queue
        self.init_queue.put(job_id)
        print(f"Job {job_id} added to init queue")

        # Wait for job to be finished
        job_details.event.wait()
        job_details.end_time = time.time()
        if job_details.results is None:
            result = {**job_details.run_results, 'resolved': job_details.eval_results['resolved'], 'critical_error': None}
        else:
            result = copy.deepcopy(job_details.results)
        if job_details.start_run_time:
            init_time_taken = job_details.start_run_time - job_details.start_time
            if job_details.start_eval_time:
                run_time_taken = job_details.start_eval_time - job_details.start_run_time
                evaluate_time_taken = job_details.end_time - job_details.start_eval_time
            else:
                run_time_taken = job_details.end_time - job_details.start_run_time
                evaluate_time_taken = 0
        else:
            init_time_taken = job_details.end_time - job_details.start_time
            run_time_taken = 0
            evaluate_time_taken = 0
        # Close runtime
        if job_details.runtime:
            job_details.runtime.close()
        # Delete job details
        del self._job_details[job_id]
        return {
            **result,
            'init_time_taken': init_time_taken,
            'run_time_taken': run_time_taken,
            'evaluate_time_taken': evaluate_time_taken,
        }

    async def _init_worker(self, wid):
        while True:
            print(f"[init-worker-{wid}] Waiting for job")
            job_id = await asyncio.to_thread(self.init_queue.get)

            # Check for stop sentinel
            if job_id == "__STOP__":
                print(f"[init-worker-{wid}] Received stop signal, exiting")
                self.init_queue.task_done()
                break

            print(f"[init-worker-{wid}] Got job {job_id}")
            job_details = self._job_details[job_id]
            self._active_init_jobs.add(job_id)
            try:
                runtime, metadata, config = await initialize_agents(
                    job_details.instance,
                    job_details.llm_config,
                    sid=job_id,
                    max_iterations=job_details.max_iterations
                    )
                job_details.runtime = runtime
                job_details.metadata = metadata
                job_details.config = config
                self.run_queue.put(job_id)
            except Exception as e:
                job_details.results = {
                    'instance_id': job_details.instance.instance_id,
                    'trajectory_id': job_details.instance.trajectory_id,
                    'git_patch': None,
                    'success': False,
                    "error": f"Error in init: {str(e)}",
                    'finish': False,
                    'messages': [],
                    'resolved': False,
                    'critical_error': 'init',
                }
                job_details.event.set()
            finally:
                self._active_init_jobs.remove(job_id)
                self.init_queue.task_done()
            await asyncio.sleep(0.1)

    async def _run_worker(self, wid):
        while True:
            print(f"[run-worker-{wid}] Waiting for job")
            job_id = await asyncio.to_thread(self.run_queue.get)

            # Check for stop sentinel
            if job_id == "__STOP__":
                print(f"[run-worker-{wid}] Received stop signal, exiting")
                self.run_queue.task_done()
                break

            job_details = self._job_details[job_id]
            print(f"[run-worker-{wid}] Got job {job_id}")
            job_details.start_run_time = time.time()
            self._active_run_jobs.add(job_id)
            try:
                run_results = await run_agent(
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

                job_details.results = {
                    'instance_id': job_details.instance.instance_id,
                    'trajectory_id': job_details.instance.trajectory_id,
                    'git_patch': None,
                    'success': False,
                    "error": f"Error in run agent: {str(e)}",
                    'finish': False,
                    'messages': [],
                    'resolved': False,
                    'critical_error': 'run',
                }
                job_details.event.set()
            finally:
                self._active_run_jobs.remove(job_id)
                self.run_queue.task_done()
            await asyncio.sleep(0.1)

    async def _eval_worker(self, wid):
        """Worker that evaluates the generated patch and produces a report."""
        # Lazy import to avoid heavy dependency at server startup
        from openhands.nvidia.swe_agent.utils import _evaluate_agent as _evaluate_patch_async

        while True:
            print(f"[eval-worker-{wid}] Waiting for job")
            job_id = await asyncio.to_thread(self.evaluate_queue.get)

            # Check for stop sentinel
            if job_id == "__STOP__":
                print(f"[eval-worker-{wid}] Received stop signal, exiting")
                self.evaluate_queue.task_done()
                break

            print(f"[eval-worker-{wid}] Got job {job_id}")
            job_details = self._job_details[job_id]
            job_details.start_eval_time = time.time()
            self._active_eval_jobs.add(job_id)
            try:
                if job_details.run_results['git_patch'] is None:
                    raise ValueError("Patch is None, cannot evaluate")
                eval_report = await _evaluate_patch_async(
                    job_details.run_results['git_patch'],
                    job_details.instance,
                    sid=f"eval_{job_id}",
                    allow_skip=self.allow_skip_eval
                    )
                # Only keep the 'report' field if present
                if isinstance(eval_report, dict) and 'report' in eval_report:
                    job_details.eval_results = eval_report['report']
                else:
                    job_details.eval_results = eval_report
                job_details.event.set()
            except Exception as e:
                job_details.results = {
                    'instance_id': job_details.instance.instance_id,
                    'trajectory_id': job_details.instance.trajectory_id,
                    'git_patch': job_details.run_results.get('git_patch', None),
                    'success': job_details.run_results.get('success', False),
                    "error": f"Error in eval: {str(e)}",
                    'finish': job_details.run_results.get('finish', False),
                    'messages': job_details.run_results.get('messages', []),
                    'resolved': False,
                    'critical_error': 'eval',
                }
                job_details.event.set()
            finally:
                self._active_eval_jobs.remove(job_id)
                self.evaluate_queue.task_done()
            await asyncio.sleep(0.1)

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
        if not hasattr(self, 'init_queue') or self.init_queue is None:
            # Server was never started or already stopped
            return

        print(f"Stopping events")
        # Signal all active jobs to complete
        for job_id in list(self._active_init_jobs):
            if job_id in self._job_details:
                self._job_details[job_id].event.set()

        for job_id in list(self._active_run_jobs):
            if job_id in self._job_details:
                self._job_details[job_id].event.set()

        print(f"Stopping queues")
        # Add sentinel values to queues to unblock workers
        for _ in range(self.max_init_workers):
            try:
                self.init_queue.put_nowait("__STOP__")
            except:
                pass

        for _ in range(self.max_run_workers):
            try:
                self.run_queue.put_nowait("__STOP__")
            except:
                pass

        for _ in range(self.max_eval_workers):
            try:
                self.evaluate_queue.put_nowait("__STOP__")
            except:
                pass

        # Forced shutdown
        for loop in self._init_workers + self._run_workers + self._eval_workers:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                pass
            finally:
                if loop:
                    loop.close()

        print(f"Shutting down executor")
        # Shutdown the executor with a timeout
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        print(f"Clearing active jobs")
        # Clear all state
        self._active_init_jobs.clear()
        self._active_run_jobs.clear()
        self._active_eval_jobs.clear()
        self._job_details.clear()
        self._init_workers.clear()
        self._run_workers.clear()
        self._eval_workers.clear()
        # Reset queues
        self.init_queue = None
        self.run_queue = None
        self.evaluate_queue = None

        print(f"Server status: {self.status()}")

    def status(self):
        """Returns the number of jobs currently being processed in both queues and workers."""
        if self.init_queue:
            init_queue_count = self.init_queue.qsize()
        else:
            init_queue_count = 0
        if self.run_queue:
            run_queue_count = self.run_queue.qsize()
        else:
            run_queue_count = 0
        if hasattr(self, 'evaluate_queue') and self.evaluate_queue:
            eval_queue_count = self.evaluate_queue.qsize()
        else:
            eval_queue_count = 0
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
            'total': init_queue_count + run_queue_count + eval_queue_count + active_init_count + active_run_count + active_eval_count,
        }

def test_server(total_jobs: int = 4, max_parallel_jobs: int = 2, allow_skip_eval: bool = False):
    import pandas as pd
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    dataset = pd.read_parquet("/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet")
    instance = dataset.iloc[0]['instance']
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    requests = []
    for i in range(total_jobs):
        cur = instance.copy(deep=True)
        cur.trajectory_id = i
        requests.append(cur)

    llm_server_address = "http://127.0.0.1:8000/v1"
    sampling_params = {
        "model": "openai/Qwen/Qwen3-8B",
        "api_key": "mykey",
        "modify_params": False,
        "log_completions": True,
        "native_tool_calling": True,
        "temperature": 0.6,
    }

    print("Starting server")
    server = OpenHandsServer(
        llm_server_addresses=[llm_server_address, llm_server_address],
        max_init_workers=max_parallel_jobs,
        max_run_workers=max_parallel_jobs,
        allow_skip_eval=allow_skip_eval,
    )
    server.start()
    print("Server started")

    print("Job submission started")

    # Process instances using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [executor.submit(server.process, inst, sampling_params) for inst in requests]
        results = [future.result() for future in futures]

    print("Job submission finished")
    #print(results)
    server.stop()
    return results

if __name__ == "__main__":
    start = time.time()
    results = test_server(total_jobs=5, max_parallel_jobs=5, allow_skip_eval=False)
    # Don't print full messages
    for result in results:
        result['messages'] = len(result['messages'])
    print(results)
    print(f"Time taken: {time.time() - start}")
