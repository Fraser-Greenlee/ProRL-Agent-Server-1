# OpenHands NVIDIA Module

The OpenHands NVIDIA module provides a high-performance, asynchronous agent processing system designed for scalable evaluation of software engineering agents. It features a registry-based architecture that supports multiple agent types and provides robust error handling and resource management.

## Architecture Overview

The module consists of several key components:

- **Registry System**: A flexible registration system for agent handlers
- **Agent Handlers**: Pluggable implementations for different agent types
- **Async Server**: High-performance asynchronous job processing

## Core Components

### 1. Registry System (`registry.py`)

The registry system provides a centralized mechanism for registering and managing agent handlers. It implements a plugin architecture where different agent types can be registered and used seamlessly.

#### Key Classes

**`JobDetails`**: A dataclass that contains all information about a processing job:
```python
@dataclass
class JobDetails:
    job_id: str | None = None
    instance: pd.Series | None = None
    max_iterations: int = 2
    llm_config: LLMConfig | None = None
    runtime: Runtime | None = None
    metadata: EvalMetadata | None = None
    config: OpenHandsConfig | None = None
    run_results: dict | None = None
    eval_results: dict | None = None
    results: dict | None = None
    event: threading.Event | None = None
    start_time: float | None = None
    start_run_time: float | None = None
    start_eval_time: float | None = None
    end_time: float | None = None
```

**`AgentHandler`**: Abstract base class defining the interface for all agent handlers:

```python
class AgentHandler(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The name identifier for this agent handler."""
        pass

    @abstractmethod
    async def init(self, instance, llm_config, sid, max_iterations) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the agent with instance and config."""
        pass

    @abstractmethod
    async def run(self, runtime, metadata, config, instance) -> dict[str, object]:
        """Run the agent with runtime and instance."""
        pass

    @abstractmethod
    async def eval(self, job_details, sid, allow_skip) -> dict[str, Any]:
        """Evaluate the agent results."""
        pass

    # Exception handlers and final result processing...
```

#### Registry Functions

The registry maintains separate function mappings for each stage of agent processing:

- **`init`**: Functions for initializing agents
  - **Input**: `instance` (pd.Series), `llm_config` (LLMConfig), `sid` (str), `max_iterations` (int)
  - **Output**: `tuple[Runtime, EvalMetadata, OpenHandsConfig]`

- **`run`**: Functions for running agents
  - **Input**: `runtime` (Runtime), `metadata` (EvalMetadata), `config` (OpenHandsConfig), `instance` (pd.Series)
  - **Output**: `dict[str, object]` containing run results

- **`eval`**: Functions for evaluating agent results
  - **Input**: `job_details` (JobDetails), `sid` (str), `allow_skip` (bool)
  - **Output**: `dict[str, Any]` containing evaluation results

- **Exception handlers**: `init_exception`, `run_exception`, `eval_exception`
  - **Input**: `job_details` (JobDetails), `exception` (Exception)
  - **Output**: `dict[str, Any]` containing error information

- **`final_result`**: Functions for processing final results
  - **Input**: `job_details` (JobDetails)
  - **Output**: `dict[str, Any]` containing final processed results

#### Registering New Agent Handlers

To add a new agent handler:

1. **Create a handler class** that inherits from `AgentHandler`:

```python
class MyAgentHandler(AgentHandler):
    @property
    def name(self) -> str:
        return "my_agent_type"

    async def init(self, instance, llm_config=None, sid=None, max_iterations=1):
        # Initialize your agent
        runtime = create_runtime(instance)
        metadata = create_metadata(instance)
        config = create_config(llm_config)
        return runtime, metadata, config

    async def run(self, runtime, metadata, config, instance):
        # Run your agent
        results = await my_agent_run_logic(runtime, metadata, config, instance)
        return {"git_patch": results.patch, "messages": results.messages}

    async def eval(self, job_details, sid=None, allow_skip=True):
        # Evaluate results
        return await my_evaluation_logic(job_details.run_results, job_details.instance)

    # Implement exception handlers and final_result...
```

2. **Register the handler**:

Register agent handler in __init__.py
```python
from openhands.nvidia.registry import register_agent_handler
register_agent_handler(MyAgentHandler())
```

### 2. SWE Agent Handler (`swe_agent/swe_agent_handler.py`)

The SWE Agent Handler is a concrete implementation of the `AgentHandler` interface for SWEBench evaluation tasks. It demonstrates the proper implementation pattern:

```python
class SweAgentHandler(AgentHandler):
    @property
    def name(self) -> str:
        return "swebench"  # This matches the data_source field in instances

    async def init(self, instance, llm_config=None, sid=None, max_iterations=1):
        return await initialize_agents(
            instance=instance,
            llm_config=llm_config,
            sid=sid,
            max_iterations=max_iterations
        )

    async def run(self, runtime, metadata, config, instance):
        return await run_agent(
            runtime=runtime,
            metadata=metadata,
            config=config,
            instance=instance
        )

    async def eval(self, job_details, sid=None, allow_skip=True):
        git_patch = job_details.run_results['git_patch'] if job_details.run_results else ''
        return await evaluate_agent(
            git_patch=git_patch,
            instance=job_details.instance,
            sid=sid,
            allow_skip=allow_skip,
        )
```

The handler delegates to utility functions in `swe_agent/utils.py` for the actual implementation details.

### 3. Async Server (`async_server.py`)

The `OpenHandsServer` provides high-performance asynchronous processing of agent jobs through a sophisticated three-stage pipeline architecture. Each job flows through distinct stages with dedicated worker pools, ensuring optimal resource utilization and fault isolation.

#### Server Architecture

The server implements a producer-consumer pattern with three specialized stages:

1. **Initialization Stage**: Creates runtime environments, metadata, and configurations
2. **Run Stage**: Executes the agent logic within the prepared runtime
3. **Evaluation Stage**: Analyzes and scores the agent's output

Each stage maintains its own:
- **Queue**: Thread-safe job queue for that stage
- **Worker Pool**: Configurable number of async workers
- **Active Job Tracking**: Set of currently processing job IDs
- **Error Handling**: Stage-specific exception handlers

#### Server Configuration

```python
server = OpenHandsServer(
    llm_server_addresses=['http://server1:8000/v1', 'http://server2:8000/v1'],
    max_init_workers=6,     # Workers for initialization stage
    max_run_workers=5,      # Workers for execution stage
    max_eval_workers=5,     # Workers for evaluation stage (defaults to max_run_workers)
    allow_skip_eval=True    # Skip evaluation if no result is generated (such as empty git-patch for swebench)
)
```

#### Detailed Pipeline Flow

**Job Submission and Setup:**
```python
def process(self, instance, sampling_params, job_id=None):
    # 1. Create JobDetails container
    job_details = JobDetails()
    job_details.job_id = job_id or self.get_unique_id(instance)
    job_details.instance = instance
    job_details.max_iterations = sampling_params.pop('max_iterations', 2)
    job_details.llm_config = self.create_llm_config(sampling_params)
    job_details.start_time = time.time()
    job_details.event = threading.Event()  # For synchronization

    # 2. Store job details and add to init queue
    self._job_details[job_id] = job_details
    self.init_queue.put(job_id)

    # 3. Wait for completion
    job_details.event.wait()
```

**Stage 1: Initialization Worker (`_init_worker`)**

The initialization stage prepares the execution environment:

```python
async def _init_worker(self, wid):
    job_id = await asyncio.to_thread(self.init_queue.get)
    job_details = self._job_details[job_id]

    # Determine agent type from instance
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _init_func = get_registered_functions('init', dataset_type)

    # Call registered init function
    runtime, metadata, config = await _init_func(
        job_details.instance,      # Input: Problem instance data
        job_details.llm_config,    # Input: LLM configuration
        sid=job_id,                # Input: Session ID for tracking
        max_iterations=job_details.max_iterations  # Input: Iteration limit
    )

    # Store results in job_details
    job_details.runtime = runtime    # Runtime environment (Docker, etc.)
    job_details.metadata = metadata  # Evaluation metadata
    job_details.config = config      # OpenHands configuration

    # Move to next stage
    self.run_queue.put(job_id)
```

**What the `init` function receives and produces:**
- **Inputs**:
  - `instance` (pd.Series): Task instance data (format depends on agent type)
  - `llm_config` (LLMConfig): LLM server configuration and parameters
  - `sid` (str): Session ID for logging and tracking
  - `max_iterations` (int): Maximum number of agent iterations allowed
- **Outputs**:
  - `runtime` (Runtime): Execution environment for the agent
  - `metadata` (EvalMetadata): Agent-specific evaluation configuration
  - `config` (OpenHandsConfig): Agent configuration including LLM settings

**Stage 2: Run Worker (`_run_worker`)**

The run stage executes the agent within the prepared environment:

```python
async def _run_worker(self, wid):
    job_id = await asyncio.to_thread(self.run_queue.get)
    job_details = self._job_details[job_id]
    job_details.start_run_time = time.time()

    # Get registered run function
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _run_func = get_registered_functions('run', dataset_type)

    # Execute agent
    run_results = await _run_func(
        job_details.runtime,   # Input: Prepared runtime environment
        job_details.metadata,  # Input: Evaluation metadata
        job_details.config,    # Input: Agent configuration
        job_details.instance,  # Input: Problem instance
    )

    # Store results and clean up runtime
    job_details.run_results = run_results  # Agent's output (git_patch, messages, etc.)

    # IMPORTANT: Runtime is closed immediately after run completes
    if job_details.runtime:
        job_details.runtime.close()
        job_details.runtime = None

    # Move to evaluation
    self.evaluate_queue.put(job_id)
```

**What the `run` function receives and produces:**
- **Inputs**:
  - `runtime` (Runtime): Active execution environment from init stage
  - `metadata` (EvalMetadata): Evaluation metadata from init stage
  - `config` (OpenHandsConfig): Agent configuration from init stage
  - `instance` (pd.Series): Original task instance
- **Outputs**:
  - `dict[str, object]` containing agent-specific results (format varies by agent type)
  - Common fields include execution traces, generated outputs, performance metrics
  - Results are stored in `job_details.run_results` for use in evaluation stage

**Stage 3: Evaluation Worker (`_eval_worker`)**

The evaluation stage assesses the agent's performance:

```python
async def _eval_worker(self, wid):
    job_id = await asyncio.to_thread(self.evaluate_queue.get)
    job_details = self._job_details[job_id]
    job_details.start_eval_time = time.time()

    # Get registered eval function
    dataset_type = getattr(job_details.instance, 'data_source', 'swebench')
    _eval_func = get_registered_functions('eval', dataset_type)

    # Evaluate results
    eval_report = await _eval_func(
        job_details,                    # Input: Complete job details
        sid=f'eval_{job_id}',          # Input: Evaluation session ID
        allow_skip=self.allow_skip_eval # Input: Skip if no patch
    )

    # Store evaluation results
    if isinstance(eval_report, dict) and 'report' in eval_report:
        job_details.eval_results = eval_report['report']
    else:
        job_details.eval_results = eval_report

    # Signal completion
    job_details.event.set()
```

**What the `eval` function receives and produces:**
- **Inputs**:
  - `job_details` (JobDetails): Complete job information including:
    - `job_details.run_results`: Agent's output from run stage
    - `job_details.instance`: Original task instance
    - `job_details.metadata`: Evaluation metadata from init stage
  - `sid` (str): Evaluation session ID
  - `allow_skip` (bool): Whether to skip evaluation based on run results
- **Outputs**:
  - `dict[str, Any]` containing agent-specific evaluation results
  - Results are stored in `job_details.eval_results` and used for final result processing

#### JobDetails State Evolution

The `JobDetails` object accumulates state as it flows through the pipeline:

**After Job Creation:**
```python
job_details.job_id = "unique_job_id"
job_details.instance = pd.Series(instance_data)
job_details.max_iterations = 2
job_details.llm_config = LLMConfig(...)
job_details.start_time = 1234567890.0
job_details.event = threading.Event()
```

**After Init Stage:**
```python
job_details.runtime = DockerRuntime(...)
job_details.metadata = EvalMetadata(...)
job_details.config = OpenHandsConfig(...)
```

**After Run Stage:**
```python
job_details.run_results = {
    # Agent-specific results - format varies by agent type
    # Common: execution outputs, traces, generated content, metrics
}
job_details.start_run_time = 1234567935.0
job_details.runtime = None  # Closed for resource management
```

**After Eval Stage:**
```python
job_details.eval_results = {
    # Agent-specific evaluation results - format varies by agent type
    # Common: scores, test results, performance metrics
}
job_details.start_eval_time = 1234567980.0
```

#### SWE Agent Handler Example

To illustrate the pipeline with a concrete example, here's how the SWE Agent Handler (for SWEBench-style tasks) uses the pipeline:

OpenHands server supports three variants of the SWE-Bench task family. Each variant is handled slightly differently in the *inference* and *evaluation* phases:

1. **SWE-Gym/SWE-Bench**
   • *Inference*: The container image name is derived from `instance_id` as `xingyaoww/sweb.eval.x86_64.<instance_id>` and executed inside an isolated sandbox. The runtime resource factor is dynamically scaled with task difficulty.
   • *Evaluation*: By default the pipeline invokes `swegym.harness.*` to apply the generated patch and run the test-suite.
   • *Extra dependency*: `pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git`

2. **SWE-Bench Multimodal**
   • *Inference*: Builds the official image path `docker.io/swebench/sweb.eval.x86_64.<repo>_1776_<issue>:latest` from `instance_id` and mounts the dataset-provided `image_assets` directory to support vision-related tests.
   • *Evaluation*: Utilises `swebench.harness` for patch application and testing; the multimodal harness automatically locates and loads the required image assets.
   • *Extra dependency*: `pip install swebench` (or install the SWE-Bench-Package above for the latest commits)

3. **R2E-Gym**
   • *Inference*: Each instance contains an explicit `docker_image` field. OpenHands pulls and executes this image directly; the runtime resource factor is fixed at 1 to minimise resource variance.
   • *Evaluation*: Uses the internal helper `_apply_patch_and_evaluate_r2egym`. The overall procedure mirrors SWE-Bench but includes a custom parser tailored to the R2E-Gym test-log format.
    In addition, the evaluator **pre-filters the patch**: any hunk that edits files already modified inside the Docker image is discarded. Consequently, patches that touch such files will be partially (or fully) ignored and may not apply cleanly.

*Tip*: If the evaluation images are not present locally, run `scripts/pull_swe_images.py` to download them in bulk.

**SWE Agent Init Stage:**
```python
# Input instance contains:
instance = pd.Series({
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "problem_statement": "Fix bug in authentication...",
    "data_source": "swebench"
})

# Output after init:
job_details.runtime = DockerRuntime(image="swe-agent:latest", ...)
job_details.metadata = EvalMetadata(reference_answer="expected_patch.diff", ...)
job_details.config = OpenHandsConfig(agent_cls="CodeActAgent", ...)
```

**SWE Agent Run Stage:**
```python
# The run function executes the agent in the Docker environment
# and produces SWEBench-specific results:
job_details.run_results = {
    "git_patch": "diff --git a/django/auth.py...",  # Generated code changes
    "messages": [                                   # Agent conversation
        {"role": "user", "content": "Please fix the authentication bug..."},
        {"role": "assistant", "content": "I'll analyze the code..."},
        # ... more conversation ...
    ],
    "metrics": {
        "tokens_used": 1500,
        "time_elapsed": 45.2,
        "iterations": 3
    },
    "test_output": "Tests passed: 25/25"
}
```

**SWE Agent Eval Stage:**
```python
# The eval function tests the generated patch against the repository
job_details.eval_results = {
    "resolved": True,                    # Whether the issue was fixed
    "test_passed": True,                 # Whether all tests pass
    "test_output": "All tests passed",   # Detailed test results
    "patch_correctness": 1.0,           # Patch quality score
    "evaluation_time": 12.3
}
```

#### Load Balancing and Resource Management

**LLM Server Load Balancing:**
```python
def create_llm_config(self, sampling_params):
    # Weighted round-robin selection
    address = self.weighted_addresses[0][1]
    self.weighted_addresses[0][0] += 1  # Increment usage counter
    heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
    return LLMConfig(base_url=address, **sampling_params)
```

**Resource Cleanup Strategy:**
- Runtimes are closed immediately after the run stage to free system resources
- Failed jobs trigger cleanup through exception handlers
- Server shutdown properly terminates all workers and cleans up remaining resources

#### Error Handling and Recovery

Each stage has dedicated exception handlers:

- **Init Exception**: Creates error result and signals completion
- **Run Exception**: Ensures runtime cleanup and creates error result
- **Eval Exception**: Handles evaluation failures gracefully

Exception handlers store error information in `job_details.results` and signal completion via `job_details.event.set()`.

#### Status Monitoring

```python
def status(self):
    return {
        'init_queue': self.init_queue.qsize(),        # Jobs waiting for init
        'run_queue': self.run_queue.qsize(),          # Jobs waiting to run
        'eval_queue': self.evaluate_queue.qsize(),    # Jobs waiting for eval
        'active_init': len(self._active_init_jobs),   # Jobs currently initializing
        'active_run': len(self._active_run_jobs),     # Jobs currently running
        'active_eval': len(self._active_eval_jobs),   # Jobs currently evaluating
        'total': total_jobs_across_all_stages
    }
```



## Usage Examples

### Basic Usage

```python
from openhands.nvidia.async_server import OpenHandsServer

# Create and start server
server = OpenHandsServer(
    llm_server_addresses=['http://localhost:8000/v1'],
    max_init_workers=4,
    max_run_workers=4
)
server.start()

# Process a job
result = server.process(instance, sampling_params)
print(f"Result: {result}")

# Stop server
server.stop()
```

### Production Workflow

```python
import asyncio
from test_server import start_server, stop_server, add_llm_server, process_request

async def production_workflow():
    # 1. Start server
    await start_server()

    # 2. Add LLM servers
    await add_llm_server('http://server1:8000/v1')
    await add_llm_server('http://server2:8000/v1')

    # 3. Process jobs concurrently
    tasks = []
    for instance in instances:
        task = process_request(instance, sampling_params)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # 4. Stop server
    await stop_server()

    return results
```

### Adding Custom Agent Types

```python
# 1. Implement handler
class CustomAgentHandler(AgentHandler):
    @property
    def name(self) -> str:
        return "custom_agent"

    # Implement required methods...

# 2. Register handler in __init__.py
from openhands.nvidia.registry import register_agent_handler
register_agent_handler(CustomAgentHandler())

# 3. Use with instances that have data_source="custom_agent"
```

## Error Handling

The system provides comprehensive error handling at multiple levels:

- **Registry Level**: `FunctionNotRegisteredError` for missing handlers
- **Server Level**: Automatic cleanup of failed jobs and resources
- **API Level**: HTTP status codes and detailed error messages
- **Job Level**: Exception handlers for each processing stage

## Performance Considerations

- **Worker Pools**: Configure worker counts based on available resources
- **LLM Servers**: Use multiple LLM servers for better load distribution
- **Timeouts**: Set appropriate timeouts to prevent resource leaks
- **Resource Cleanup**: The system automatically closes runtimes and cleans up resources
