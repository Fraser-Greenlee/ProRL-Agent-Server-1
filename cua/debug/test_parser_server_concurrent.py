import os
import time
import asyncio
import httpx
from pathlib import Path


async def send_single_request(client: httpx.AsyncClient, server_url: str, image_bytes: bytes, filename: str,
                              req_id: int):
    """
    Sends a single async request. Returns result if success, None if failed.
    """
    try:
        # Construct the multipart form data exactly like the requests library
        files = {"file": (filename, image_bytes, "image/png")}

        # Send POST request (timeout set higher to allow queueing time on server)
        response = await client.post(server_url, files=files, timeout=60.0)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Req {req_id}: Server Error {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Req {req_id}: Error {e}")
        return None


async def run_concurrent_test(server_url: str, image_path: str, concurrency: int = 512):
    print(f"--- Starting Load Test ---")
    print(f"Target:      {server_url}")
    print(f"Concurrency: {concurrency} requests")

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # 1. Load image into memory ONCE (avoid disk bottleneck)
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Image loaded ({len(image_bytes) / 1024:.2f} KB). Firing requests...")

    # 2. Configure Client with no connection limits
    # Standard limits are too low for 512 concurrent connections
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)

    async with httpx.AsyncClient(limits=limits) as client:
        start_time = time.time()

        # 3. Create all tasks instantly
        tasks = [
            send_single_request(client, server_url, image_bytes, filename, i)
            for i in range(concurrency)
        ]

        # 4. Fire them all at once and wait for completion
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

    # 5. Calculate Stats
    successful_responses = [r for r in results if r is not None]
    success_count = len(successful_responses)
    fail_count = concurrency - success_count
    throughput = success_count / total_time

    print(f"\n--- Benchmark Results ---")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Successful:     {success_count}/{concurrency}")
    print(f"Failed:         {fail_count}")
    print(f"Throughput:     {throughput:.2f} img/sec 🚀")

    # Optional: Inspect one result to ensure correctness
    if successful_responses:
        print(f"Sample Result: {successful_responses[0]['count']} boxes detected in first response.")


if __name__ == "__main__":
    # Configuration
    SERVER_URL = "http://pool0-01436:8000/parse"
    IMAGE_PATH = "../../examples/screenshots/click.png"
    CONCURRENT_REQUESTS = 512

    # Run the async main loop
    asyncio.run(run_concurrent_test(SERVER_URL, IMAGE_PATH, CONCURRENT_REQUESTS))
