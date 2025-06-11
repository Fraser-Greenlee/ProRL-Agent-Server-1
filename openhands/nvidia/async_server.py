import asyncio
from openhands.nvidia.swe_agent.utils import initialize_agents, run_agent
from openhands.core.config.llm_config import LLMConfig
from typing import List
import heapq

def get_job_id(instance):
    return f"swebench_{instance.instance_id}_{instance.trajectory_id}"

class AsyncOpenHandsServer:
    def __init__(self, llm_server_addresses: List[str] = [], max_init_workers:int = 6, max_run_workers:int = 5):
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.init_queue = asyncio.Queue(maxsize=max_init_workers)
        self.run_queue = asyncio.Queue(maxsize=max_run_workers)
        self._init_workers = []
        self._run_workers = []
        self._results = {}  # jobid -> Future
        self._active_init_jobs = set()  # Track jobs being initialized
        self._active_run_jobs = set()   # Track jobs being run

        self.weighted_addresses = [[0, address] for address in llm_server_addresses]
        heapq.heapify(self.weighted_addresses)

    def add_llm_server_address(self, llm_server_address: str):
        heapq.heappush(self.weighted_addresses, [0, llm_server_address])

    def create_llm_config(self, sampling_params):
        if len(self.llm_server_addresses) == 0:
            raise ValueError("No LLM server addresses added")
        
        address = self.weighted_addresses[0][1]
        self.weighted_addresses[0][0] += 1
        heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        llm_config = LLMConfig(
            base_url = address,
            **sampling_params
        )
        return llm_config
    
    async def process(self, instance, llm_server_address, sampling_params):
        if len(self.llm_server_addresses) == 0:
            raise ValueError("No LLM server addresses added")
        
        llm_config = self.create_llm_config(sampling_params, llm_server_address)
        job_id = get_job_id(instance)
        if job_id in self._results:
            raise ValueError(f"Job {job_id} already added")
        future = asyncio.get_running_loop().create_future()
        self._results[job_id] = future
        await self.init_queue.put((instance, llm_config))
        results = await future
        del self._results[job_id]
        return results

    async def _init_worker(self, wid):
        while True:
            instance, llm_config = await self.init_queue.get()
            job_id = get_job_id(instance)
            print(f"[init-worker-{wid}] Got job {job_id}")
            self._active_init_jobs.add(job_id)
            try:
                initialized_results = await initialize_agents(instance, llm_config)
                asyncio.create_task(self.run_queue.put(initialized_results)) # Fire-and-forget
            except Exception as e:
                future = self._results.get(job_id)
                if future and not future.done():
                    future.set_exception(e)
            finally:
                self._active_init_jobs.remove(job_id)
                self.init_queue.task_done()
            asyncio.sleep(0.1)

    async def _run_worker(self, wid):
        while True:
            runtime, metadata, config, instance = await self.run_queue.get()
            job_id = get_job_id(instance)
            print(f"[run-worker-{wid}] Got job {job_id}")
            self._active_run_jobs.add(job_id)
            try:
                results = await run_agent(runtime, metadata, config, instance)
                future = self._results.get(job_id)
                if future and not future.done():
                    future.set_result(results)
            except Exception as e:
                import pdb; pdb.set_trace()
                future = self._results.get(job_id)
                if future and not future.done():
                    future.set_exception(e)
            finally:
                self._active_run_jobs.remove(job_id)
                self.run_queue.task_done()
            asyncio.sleep(0.1)

    async def start(self):
        self._init_workers = [asyncio.create_task(self._init_worker(i)) for i in range(self.max_init_workers)]
        self._run_workers = [asyncio.create_task(self._run_worker(i)) for i in range(self.max_run_workers)]

    async def stop(self):
        for task in self._init_workers + self._run_workers:
            task.cancel()

    def status(self):
        """Returns the number of jobs currently being processed in both queues and workers."""
        init_queue_count = self.init_queue.qsize()
        run_queue_count = self.run_queue.qsize()
        active_init_count = len(self._active_init_jobs)
        active_run_count = len(self._active_run_jobs)
        return {
            'init_queue': init_queue_count,
            'run_queue': run_queue_count,
            'active_init': active_init_count,
            'active_run': active_run_count,
            'total': init_queue_count + run_queue_count + active_init_count + active_run_count
        }

async def test_server():
    import pandas as pd
    import numpy as np

    dataset = pd.read_parquet("/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet")
    instance = dataset.iloc[0]['instance']
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    requests = []
    for i in range(2):
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

    futures = []
    server = AsyncOpenHandsServer(llm_server_addresses=[llm_server_address, llm_server_address], max_init_workers=1, max_run_workers=1)
    await server.start()
    for instance in requests:
        future =  server.process(instance, llm_server_address, sampling_params)
        futures.append(future)
    results = await asyncio.gather(*futures)
    await server.stop()
    return results

if __name__ == "__main__":
    results = asyncio.run(test_server())
    print(results)