# Scripts Directory

This directory contains utility scripts for OpenHands operations, including server management and container image handling.

## Scripts Overview

### `start_server.py`

A FastAPI-based asynchronous server for OpenHands that provides REST API endpoints for managing and processing requests.

**Purpose:**
- Manages OpenHands server lifecycle (start/stop/status)
- Handles LLM server configuration
- Processes evaluation requests with timeout management
- Provides thread pool management for concurrent processing

**Key Features:**
- REST API endpoints for server control
- Configurable worker pools for initialization and execution
- Timeout management for job processing
- Singularity job cleanup
- Error handling with appropriate HTTP status codes

**Usage:**
```bash
export LOG_LEVEL=ERROR
export DEBUG=False
python start_server.py [OPTIONS]
```

**Options:**
- `--max-init-workers`: Maximum number of initialization workers (default: 64)
- `--max-run-workers`: Maximum number of run workers (default: 64)
- `--timeout`: Global timeout for job processing in seconds (default: 300)
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8006)
- `--allow-skip-eval`: Allow skipping evaluation if git_patch is None or empty (default: True)
- `--reward-server-ip`: Reward server ip for math/code/reasoning gym (default: []). Example: `--reward_server_ip cpu-0004`.

**API Endpoints:**

#### `POST /start`
- **Purpose:** Start the OpenHands server
- **Input:** No request body required
- **Output:** `{"status": "Server started successfully"}`
- **Notes:** Automatically kills existing Singularity jobs before starting

#### `POST /stop`
- **Purpose:** Stop the OpenHands server
- **Input:** No request body required
- **Output:** `{"status": "Server stopped successfully"}` or `{"status": "Force killed all singularity jobs."}`
- **Notes:** Gracefully stops server and cleans up Singularity jobs

#### `GET /status`
- **Purpose:** Get current server status
- **Input:** No request body required
- **Output:** Server status information including worker counts and queue status

#### `POST /add_llm_server`
- **Purpose:** Add LLM server address for load balancing
- **Input:**
  ```json
  {
    "address": "http://llm-server:port/v1"
  }
  ```
- **Output:** `{"status": "Added LLM server address: <address>"}`
- **Notes:** Must be called to setup LLM addresses before sending evaluation requests

#### `POST /clear_llm_server`
- **Purpose:** Clear all configured LLM server addresses
- **Input:** No request body required
- **Output:** `{"status": "Cleared all LLM server addresses"}`
- **Notes:** Removes all LLM server addresses from the configuration. Useful for resetting the server configuration.

#### `POST /process`
- **Purpose:** Process evaluation requests with timeout management
- **Input Request Format:**
  ```json
  {
    "instance": {
      "instance_id": "string",
      "trajectory_id": "int",
      "data_source": "string", // Must match registered type in openhands.nvidia.async_server (default: "swebench")
      // ... other benchmark-specific fields
    },
    "sampling_params": {
      "model": "string",              // Model name (e.g., "hosted_vllm/Qwen/Qwen3-8B")
      "api_key": "string",            // API key (use any string if not required, e.g., "mykey")
      "modify_params": false,         // Always set to false for now
      "log_completions": "bool",      // Whether to log LLM completions (recommend: false)
      "native_tool_calling": "bool",  // If LLM supports tool calling (recommend: true for Qwen3)
      "temperature": "float",         // Sampling temperature
      "top_p": "float",              // Sampling top_p
      "max_output_tokens": "int",    // max output tokens for each iteration. max_model_len is handled by LLM server not openhands server (recommend: 4096)
      "max_iteration": "int",          // Maximum iterations for OpenHands agent
      // ... additional sampling params supported by LLMConfig in openhands.core.config.llm_config
    }
  }
  ```

- **Output Response Format:**

  **Generic Response:**
  ```json
  {
    "instance_id": "string",
    "trajectory_id": "string"
    // ... other benchmark-specific fields
  }
  ```

  **SWE-Bench Specific Response:**
  ```json
  {
    "instance_id": "string",
    "trajectory_id": "string",
    "critical_error": "string",  // Pipeline error stage (null indicates success)
    "resolved": "bool",          // Whether agent resolved the issue

    // Fields included for successful pipeline runs:
    "git_patch": "string",       // Agent-submitted patch
    "success": "bool",          // Whether OpenHands agent state is not fatal error
    "error": "string",          // Error messages from OpenHands and pipeline
    "finish": "bool"            // Whether OpenHands agent reached exit state
  }
  ```

- **Important Notes:**
  - Requests are automatically load-balanced across configured LLM servers in round-robin fashion
  - The `data_source` field must match registered benchmark types in the server
  - Response format varies by benchmark type (SWE-Bench example shown above). Will update to rewards in future.

**Recommended Usage Patterns:**

#### For RL Training Inference (Recommended Workflow)

**One-time Setup:**
1. **Setup LLM Servers:** Use `POST /clear_llm_server` then `POST /add_llm_server` to configure all LLM endpoint addresses (add all server addresses before training)

**For each batch of inference during training:**
1. **Start Server:** `POST /start` to initialize the server
2. **Submit Concurrent Requests:** Post async process requests via `POST /process`
   - Concurrent requests are automatically balanced in round-robin fashion across LLM endpoints
   - **Important:** Only submit slightly more requests than your `max-run-workers` can handle
   - **Example:** For `max-run-workers=64`, submit ~68 concurrent requests (4 additional)
   - This ensures the server isn't overwhelmed while maintaining a small queue for immediate processing
3. **Stop Server:** `POST /stop` to refresh server status and ensure health

#### Recommended Settings
- **Worker Configuration:**
  - `max-init-workers` should be ≥ `max-run-workers` (recommended: keep them equal)
  - `max-run-workers` should be carefully calculated based on your LLM infrastructure

- **Calculating max-run-workers:**
  - **Formula:** `(Number of LLM servers) × (Concurrent requests per server)`
  - **Example:** 4 LLM servers × 4 concurrent requests each = 16 max-run-workers
  - Balance concurrent requests to achieve high GPU utilization without oversubscribing LLM service or CPU

- **Concurrency Guidelines:**
  - **Critical:** Only submit slightly more requests than your `max-run-workers` can handle
  - **Optimal Ratio:** For `max-run-workers=16`, submit ~18 concurrent requests (2 additional buffer)
  - **Purpose:** Ensures the server isn't overwhelmed while maintaining a small queue for immediate processing
  - **Monitoring:** Track LLM server capacity and response times
  - **Tuning:** Adjust concurrent request limits based on server performance
  - **Resource Considerations:** Account for memory and CPU constraints on both client and server sides

### `pull_swe_images.py`

A utility script for collecting and building Singularity container images from SWE-Bench/SWE-Bench multimodal datasets.

**Purpose:**
- Extracts Docker image requirements from SWE-Bench parquet files
- Converts Docker images to Singularity .sif format
- Manages batch processing of multiple images
- Handles caching and temporary directory management

**Key Features:**
- Parquet file parsing for instance IDs
- Docker to Singularity conversion pipeline
- Configurable image prefixes and naming
- Batch processing with start/end index support
- Logging and progress tracking
- Automatic cache directory management

**Usage:**
First specific the local Directory of singularities images:
```bash
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/
OpenHands_internal/singularity_images
```
then run
```bash
python pull_swe_images.py [OPTIONS]
```
It would better to copy the images from `/lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/
OpenHands_internal/singularity_images` to your own path.

**Required Arguments:**
- `--parquet-file`: Path to a SWE-Bench parquet file (train/validation etc.)

**Optional Arguments:**
- `--prefix`: Override Docker image namespace prefix
- `--dest-dir`: Directory to store .sif images (default: `<workspace>/singularity_images`)
- `--temp-base`: Base directory for temporary build folders (default: `<dest>/temp_dif`)
- `--start-index`: 1-based index of first image to process (default: 1)
- `--end-index`: 1-based index of last image to process (inclusive)
- `--log-name`: Capture combined stdout/stderr to a log file under `images_process/log/`

**Environment Variables:**
- `EVAL_DOCKER_IMAGE_PREFIX`: Default Docker image prefix
- `DEST_DIR`: Override destination directory
- `TEMP_BASE`: Override temporary build directory

### `run_swe.py`

A convenience script for bulk evaluation of SWE-Bench instances using the OpenHands asynchronous server. It launches the server, streams evaluation requests, and collects the results in a single command-line call.

**Purpose:**
- Automates end-to-end evaluation on SWE-Bench parquet datasets
- Spawns `start_server.py` as a background subprocess and waits until it is ready
- Sends instances to the server with adjustable concurrency and sampling parameters
- Supports round-robin load balancing across multiple LLM endpoints
- Persists results to a newline-delimited JSON (`.jsonl`) file
- Provides graceful shutdown and error handling for interrupted runs

**Key Features:**
- Asynchronous HTTP client built on `aiohttp` with semaphore-controlled concurrency
- Simple progress logging every 50 completed instances
- Flexible sampling parameter overrides via JSON string
- Optional sub-sampling of the dataset for quick testing
- Automatic cleanup of server process on completion or error

**Usage:**
```bash
python run_swe.py [OPTIONS]
```

**Optional Arguments:**
- `--dataset-path`: Path to a SWE-Bench parquet file (default shown in script)
- `--output`: Path to save evaluation results (`.jsonl`) (default: `eval_results.jsonl`)
- `--llm-addresses`: One or more LLM HTTP endpoint base URLs (default: `http://127.0.0.1:8000/v1`)
- `--host`: Host where the OpenHands async server will listen (default: `localhost`)
- `--port`: Port for the async server (default: `8006`)
- `--concurrency`: Maximum concurrent evaluation requests (default: `32`)
- `--num-instances`: Limit number of instances to evaluate (useful for debugging)
- `--sampling-params`: JSON string merged into default sampling parameters

**Output Format:**
Each line of the output file is a JSON object mirroring the response schema documented in `start_server.py`, including fields such as `instance_id`, `trajectory_id`, and benchmark-specific keys (e.g., `resolved`, `critical_error`).

**Example:**
```bash
python run_swe.py \
  --dataset-path /path/to/train.parquet \
  --output swe_results.jsonl \
  --llm-addresses http://10.0.0.2:8000/v1 http://10.0.0.3:8000/v1 \
  --concurrency 64 \
  --sampling-params '{"temperature": 0.3, "top_p": 0.95}'
```

### `prepare_data.py`

Utility for merging and standardising benchmark parquet datasets (SWE-Bench, SWE-Bench multimodal, R2E-Gym) into a single unified file.

**Purpose:**
- Recursively scans the provided directory for parquet shards.
- Detects dataset type based on the presence of signature columns and applies the corresponding per-row preprocessing:
  - `docker_image`  → R2E-Gym (`pre_process_r2egym_instance`)
  - `image_assets`  → SWE-Bench multimodal (`pre_process_swebench_mm_instance`)
  - Otherwise       → SWE-Bench (`pre_process_swebench_instance`)
- Converts nested NumPy arrays to plain Python lists so the data can be serialised to parquet.
- Adds helper columns (`data_kind`, `data_source`, `instance_id`, etc.) to keep provenance after merging.
- Concatenates all processed rows and writes a single `merged.parquet` into the same directory.

**Key Features:**
- Automatic dataset-type detection with minimal configuration.
- Pluggable preprocessing helpers if your schema evolves.
- Handles heterogeneous schemas by performing an outer-style concat across dataframes.
- Optional normalisation step to ensure parquet-compatible types (lists → JSON strings, etc.).

**Usage:**
```bash
python prepare_data.py --data-dir /path/to/parquet_directory
```

**Arguments:**
- `--data-dir` **(required)**: Directory containing one or more parquet files. The search is recursive.

**Output:**
- `merged.parquet` written to the same `--data-dir` directory.

**Example:**
```bash
python prepare_data.py --data-dir data/merged
```

---

## Data Curation

- **Train**:
1. Swe-Gym - SkyRL-v0-293-data: gpt-4o-2024-08-06 or claude-3-5-sonnet-20241022 can answers correctly, provided by SWE-Gym.
Image-path: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/singularity_images
Data-path: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/train.parquet

2. Swe-bench Multimodal - DevSet:
Image-path: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/singularity_images
Data-path: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/swe-bench-multimodal/data/train.parquet

3.R2E-Gym-Lite - Devset:
Image-path: TBD
Data-path: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/r2egym/data

- **Test**:
1. Swe-bench verified
2. Swe-bench multimodal
