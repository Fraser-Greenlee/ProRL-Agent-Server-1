# OpenHands Profiling Scripts

This directory contains scripts for profiling OpenHands server performance, including latency testing and batch processing evaluation.

## Overview

The profiling process involves four main steps:
1. **Launch LLM Servers**: Start multiple VLLM servers for load balancing
2. **Profile Latency**: Test individual request latency to determine optimal timeouts
3. **Start OpenHands Server**: Configure and start the main OpenHands server
4. **Profile Batch Processing**: Test performance with batch requests

## Prerequisites

- CUDA-enabled GPUs (the scripts are configured for 8 GPUs)
- Python environment with required dependencies
- Access to the Qwen3-14B model
- SWE-Bench dataset for testing

## Step 1: Launch LLM Servers

```bash
./launch_llm_servers.sh
```

This script:
- Sets up the Python environment and installs dependencies
- Launches 4 VLLM servers on ports 8000-8003
- Each server uses 2 GPUs with tensor parallelism
- Configures the servers with Qwen3-14B model, tool calling, and reasoning capabilities
- Waits for all servers to become ready before completing

The script automatically distributes GPUs across servers:
- Server 1 (port 8000): GPUs 0,1
- Server 2 (port 8001): GPUs 2,3
- Server 3 (port 8002): GPUs 4,5
- Server 4 (port 8003): GPUs 6,7

## Step 2: Profile Latency

```bash
python profile_latency.py
```

This script:
- Tests latency of individual OpenHands instances
- Processes 16 parallel jobs by default
- Uses SWE-Bench dataset for realistic testing
- Outputs timing information and saves results to JSON
- **Use the printed latency results to choose a proper timeout value for Step 3**

Key metrics provided:
- Total processing time
- Individual job completion times
- Average latency per request
- Server response characteristics

Example output:
```
Time taken: 45.2 seconds
All tests passed!
```

Use this timing information to set appropriate timeout values.

## Step 3: Start OpenHands Server

```bash
python ../start_server.py
```

This script starts the main OpenHands server with configurable parameters. Try different configurations to optimize performance:

### Key Configuration Options:

- `--max-init-workers`: Maximum number of initialization workers (default: 6)
- `--max-run-workers`: Maximum number of execution workers (default: 5)
- `--timeout`: Request timeout in seconds (default: 300)

### Example Configurations:

```bash
# High throughput configuration
python ../start_server.py --max-init-workers 32 --max-run-workers 32 --timeout 500
```

### Tuning Guidelines:

- **max-init-workers**: Increase for faster job initialization (limited by CPU cores)
- **max-run-workers**: Increase for higher concurrent execution (limited by GPU memory)
- **timeout**: Set based on Step 2 results

The server will start on `localhost:8000` by default and provide REST API endpoints for processing requests.

## Step 4: Profile Batch Processing

```bash
python profile_batch.py
```

This script:
- Tests batch processing with 256 requests by default
- Evaluates server performance under load
- Distributes requests across available OpenHands servers
- Provides detailed timing and throughput metrics

Key features:
- Concurrent request processing
- Automatic load balancing across servers
- Comprehensive performance reporting
- Error handling and retry logic

Expected output includes:
- Total batch processing time
- Requests per second
- Success/failure rates
- Server utilization statistics

## Files Description

- `launch_llm_servers.sh`: Launches multiple VLLM servers for load balancing
- `profile_latency.py`: Individual request latency testing
- `profile_batch.py`: Batch processing performance testing
- `../start_server.py`: Main OpenHands server with configurable parameters

## Troubleshooting

### Common Issues:

1. **GPU Memory Issues**: Reduce tensor-parallel-size or number of servers
2. **Timeout Errors**: Increase timeout values based on latency profiling results
3. **Server Startup Failures**: Check GPU availability and model access
4. **Port Conflicts**: Ensure ports 8000-8003 are available

### Performance Optimization:

1. **For High Throughput**: Increase worker counts and server replicas
2. **For Low Latency**: Reduce worker counts and optimize timeout values
3. **For Reliability**: Increase timeout values and enable retry logic

## Monitoring

Monitor the following metrics during profiling:
- GPU utilization and memory usage
- CPU utilization
- Network latency
- Request success rates
- Memory consumption

Use tools like `nvidia-smi`, `htop`, and server status endpoints for real-time monitoring.
You can also post request to server to check status: `curl -X GET http://127.0.0.1:8006/status`
During monitoring, it is best to checkout GPU utilization and also the amount of "active_run" matches expectations.
