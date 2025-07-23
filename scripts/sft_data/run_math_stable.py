import sys
import os
import argparse
import pandas as pd
import json
import time
from pathlib import Path
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../profile"))
print(f"Adding path: {script_dir}")
sys.path.append(script_dir)
from profile_batch import OpenHandsBatchProcessor
import asyncio
import subprocess
import aiohttp
import numpy as np
import signal

from openhands.nvidia.utils import get_instance_id

DEFAULT_SAMPLING_PARAMS = {
    "model": "hosted_vllm/Qwen/Qwen3-8B",
    "api_key": "mykey",
    "modify_params": False,
    "log_completions": False,
    "native_tool_calling": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "max_iterations": 35,
}

def _url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"

async def evaluate(args):
    df = pd.read_parquet(args.dataset_path)
    if args.num_instances:
        df = df.head(args.num_instances)


    instances: list[dict] = []
    for idx, row in df.iterrows():
        s = pd.Series(row).apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        s["trajectory_id"] = 0
        s = s.to_dict()
        s["instance_id"] = get_instance_id(s)
        instances.append(s)

    params = DEFAULT_SAMPLING_PARAMS.copy()
    if args.sampling_params:
        params.update(json.loads(args.sampling_params))

    if args.disable_thinking:
        params["enable_thinking"] = False

    print(f"Sampling params: {params}")
    # Configure OpenHands servers
    openhands_urls = [
        f'http://127.0.0.1:{args.port}',
    ]

    server_addresses = args.llm_addresses

    # Create processor
    processor = OpenHandsBatchProcessor(
        openhands_base_urls=openhands_urls,
        openhands_num_workers=args.concurrency,
        server_addresses=server_addresses,
        strict=False,
        save_file=args.output,
        log_freq=60, # log every 60 seconds
    )

    processor.set_sampling_params(**params)

    print(f"Sampling params: {params}")

    if args.reset_batch_size is None:
        results = await processor.generate_sequences(instances)
    else:
        # split instances into batches
        batches = [instances[i:i+args.reset_batch_size] for i in range(0, len(instances), args.reset_batch_size)]
        results = []
        for batch in batches:
            results.extend(await processor.generate_sequences(batch))

def parse_args():
    p = argparse.ArgumentParser("Simple bulk evaluation with OpenHands async server")
    # for code dataset use: /lustre/fsw/portfolios/nvr/users/mingjiel/data/eurus2-rl-data/train_code.parquet
    p.add_argument("--dataset-path", default="/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/train.parquet")
    p.add_argument("--output", default="eval_results.jsonl")
    p.add_argument("--llm-addresses", nargs="+", default=["http://127.0.0.1:8000/v1"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--num-instances", type=int)
    # need to launch reward server. Then pass in the ip address of the reward server.
    p.add_argument("--reward-server-ip", type=str, nargs="+", default=[])
    # Turn thinking off if using code dataset.
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument(
        "--sampling-params",
        default="",
        help="JSON string to merge into default sampling params",
    )
    p.add_argument("--timeout", type=int, default=1000)
    p.add_argument("--reset-batch-size", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    start_server_path = Path(__file__).parent.with_name("start_server.py")
    cmd = [sys.executable, str(start_server_path), "--port", str(args.port), "--reward-server-ip", *args.reward_server_ip, "--timeout", str(args.timeout)]
    server_proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    async def _wait_until_ready(timeout: int = 15):


        url = _url(args.host, args.port, "/status")
        start_t = time.time()
        while time.time() - start_t < timeout:
            if server_proc.poll() is not None:
                raise RuntimeError("start_server.py exited unexpectedly")
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(url) as resp:
                        if resp.status in {200, 503}:
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Async server not ready after {timeout}s on {url}")

    asyncio.run(_wait_until_ready())

    try:
        asyncio.run(evaluate(args))
    finally:
        try:
            if hasattr(os, "killpg") and server_proc.poll() is None:
                os.killpg(server_proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            server_proc.terminate()
        except Exception:
            pass
        try:
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()


