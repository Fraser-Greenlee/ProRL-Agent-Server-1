# ProRL Agent Rollout Protocol

A lightweight and ultra-flexible protocol for running **massively parallel LLM agent rollouts**
on **ANY** agent harness. It sits between your agent harness (Claude Code, OpenCode, Codex, Aider, OpenHands ...)
and your model, transparently proxying any supported API format while capturing
every completion for trajectory construction and reward assignment.

<p align="center">
  <img src="assets/arp.svg" alt="Agent Rollout Protocol logo" width="800"/>
</p>


---

## Core Features

### Rollout as a Service

Submit a single task request and the rollout server handles the rest — dispatching *N* parallel sessions across a pool of gateway nodes, load-balancing, health tracking and collecting async trajectories through gateway node callbacks.

### Framework Agnostic Agent Rollout

With the unique proxy-rerouting design, ARP can be used to rollout **ANY** agent harness and environments by subscribing to internal requests.
Register your agent harnesses through a structured `AgentSpec` with named harness types, model names, MCP servers, and skills. Or use any of the built-in harnesses:

| Harness | API Wire Format | Agent CLI |
|---------|----------------|-----------|
| `opencode` | OpenAI Chat | OpenCode |
| `claude_code` | Anthropic Messages | Claude Code |
| `codex` | OpenAI Responses | Codex CLI |
| `aider` | OpenAI Chat / Anthropic | Aider |
| `gemini_cli` | Google Generative AI | Gemini CLI |
| `openhands_sdk` | OpenAI Chat | OpenHands SDK |
| `qwen_code` | OpenAI Chat | Qwen Code |
| `swe_agent` | OpenAI Chat | SWE-Agent |
| `shell` | Any | Custom shell command |

Custom harnesses can be registered via `import_path` or the `custom_shell` escape hatch.

### Async Stage Pipelining

Inspired by [ProRL Agent Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) and SkyRL, `INIT` submits prepared runtimes to `READY` buffer for async `RUN` collection, sending time-consuming CPU-bound docker / apptainer initializations to the background, and maximizing system GPU utilization.

<p align="center">
  <img src="assets/skyrl.png" alt="Async Pipeline" width="800"/>
</p>

### Flexible Trajectory Construction

Completion records are assembled into structured traces via pluggable builders:

| Builder | Behavior |
|---------|----------|
| `all_records` | One trace per completion — simple and lossless |
| `prefix_merging` | Merges consecutive completions into longer multi-turn traces when each prompt is exactly the prior prompt + response; splits on context compaction |

Custom trajectory builders can be registered through a plugin registry (`module:ClassName`).

---

## Quick Start

See [`examples/calculator/README.md`](examples/calculator/README.md) for the
localhost calculator harness matrix, or
[`examples/swegym/README.md`](examples/swegym/README.md) for the curated
10-task SWE-Gym benchmark example with per-instance runtime images.

---

## Request Interface

```json
{
  "task_id": "example-task-001",
  "instruction": "Write a calculator and save it as calculator.py",
  "num_rollouts": 16,
  "timeout_seconds": 900,
  "runtime": {
    "backend": "docker",
    "image": "arp-localhost-opencode:latest",
    "prepare": [
      {"type": "upload_file", "source": "/host/path/test.py", "target": "/arp/session/workspace/test.py"},
      {"type": "exec", "command": "cd /arp/session/workspace && git init", "timeout_sec": 30}
    ],
    "workdir": "/arp/session/workspace",
    "network": "host"
  },
  "agent": {
    "harness": "opencode",
    "model_name": "openai/MiniMaxAI/MiniMax-M2.5",
    "timeout": 300
  },
  "builder": {"strategy": "prefix_merging"},
  "evaluator": {
    "strategy": "git_diff_patch",
    "config": {"repo_dir": "/arp/session/workspace"},
    "refresh_runtime": true
  }
}
```

---

## Architecture

```
src/
  runtime/          # Container runtime abstraction (Docker, Singularity)
    base.py         #   BaseRuntime: start/stop/exec/upload/download
    docker.py       #   DockerRuntime
    singularity.py  #   SingularityRuntime
    models.py       #   RuntimeSpec, ExecInput, ExecResult, PrepareAction
    factory.py      #   create_runtime()

  integration/      # Agent harness framework
    base.py         #   BaseHarness: setup/run_steps/cleanup_steps/postprocess
    factory.py      #   create_harness()
    models.py       #   AgentSpec, MCPServerSpec, AgentRunResult
    harnesses/      #   Built-in harness implementations
      opencode.py, claude_code.py, codex.py, aider.py,
      gemini_cli.py, qwen_code.py, openhands_sdk.py,
      swe_agent.py, shell.py

  gateway/          # FastAPI proxy & node execution manager
    server.py       #   Proxy routes, API detection & transformation
    node.py         #   INIT/READY/RUN/POSTRUN lifecycle
    dispatcher.py   #   Stage-isolated worker pools
    config.py       #   YAML + env var configuration

  rollout/          # Dispatch & collection pipeline
    server.py       #   Rollout HTTP server
    pipeline.py     #   Dispatch + result collection
    models.py       #   TaskRequest, SessionDispatchRequest

  trajectory/       # Trajectory building & evaluation
    models.py       #   EvaluatorSpec, Trajectory, Trace
    builder/        #   all_records, prefix_merging
    evaluator/      #   git_diff_patch, status_outcome
```

---

## Auto-Detected Proxies

The gateway auto-detects the incoming API format and transforms it to OpenAI Chat for vLLM:

| API Type | Detection | Transformer |
|----------|-----------|-------------|
| `anthropic` | `/v1/messages`, `anthropic-version` header | Anthropic Messages ↔ OpenAI Chat |
| `openai_chat` | `/v1/chat/completions` | Passthrough |
| `openai_responses` | `/v1/responses` | OpenAI Responses ↔ OpenAI Chat |
| `google` | `generateContent` in path, `x-goog-api-key` header | Google Generative AI ↔ OpenAI Chat |
