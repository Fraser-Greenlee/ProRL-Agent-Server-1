# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProRL-Agent-Server is an asynchronous multi-turn rollout infrastructure for RL agent training and evaluation, built on top of OpenHands. It decouples RL training from rollout execution, enabling high-concurrency evaluation with pluggable agent handlers and LLM load balancing.

## Common Commands

### Build & Setup
```bash
INSTALL_DOCKER=0 make -f Makefile.singularity build  # For Singularity-based environments
make build                                            # Standard build (requires Docker)
poetry install                                        # Install Python dependencies only
```

### Linting
```bash
make lint                                              # Run pre-commit hooks on openhands/evaluation/tests
make lint-scripts                                      # Lint scripts/ directory
poetry run pre-commit run --files <path> --show-diff-on-failure --config ./dev_config/python/.pre-commit-config.yaml
```

### Testing
```bash
# Run a specific test file
TEST_RUNTIME=singularity RUN_AS_OPENHANDS=False PYTHONPATH='.' pytest tests/runtime/test_browsing.py -v -s

# Tests live in tests/ with nvidia-specific tests in tests/nvidia/
```

### Running the Server
```bash
# Start the async evaluation server
python scripts/start_server.py --host 0.0.0.0 --port 8006 --max-init-workers 64 --max-run-workers 64 --timeout 300

# Start the FastAPI backend (WebSocket UI server)
make start-backend
```

### Key Workflows
```bash
# Pull Singularity images for SWE-Bench evaluation
python scripts/pull_swe_images.py --parquet-file train.parquet

# Run bulk SWE-Bench evaluation
python scripts/run_swe.py --dataset-path train.parquet --llm-addresses http://host:8000/v1 --concurrency 64

# Merge/standardize parquet datasets
python scripts/prepare_data.py --data-dir /path/to/datasets
```

## Architecture

### Three-Stage Async Pipeline (`openhands/nvidia/`)
The core innovation is a three-stage pipeline with separate worker pools:
1. **Init** — Prepare runtime, metadata, config for each job
2. **Run** — Execute the agent loop, generate LLM interactions
3. **Eval** — Score/evaluate results (e.g., SWE-Bench test execution)

Implemented in `async_server.py` (thread-based) and `async_server_process.py` (process-based). Each stage has its own exception handler to prevent cascading failures.

### Handler Registration System (`openhands/nvidia/registry.py`)
Agent handlers implement the `AgentHandler` ABC and are registered at import time in `openhands/nvidia/__init__.py`. Instances are routed to handlers by `instance['data_source']` field. Name mappings allow dataset aliases (e.g., `codecontests` → `deepcoder`).

Current handlers:
- `SweAgentHandler` — SWE-Bench tasks
- `MathHandler` / `CodeHandler` — Math/code reasoning
- `ProRLHandler` — ProRL multi-task (reasoning mode)
- `STEMHandler` — STEM tasks
- `GuiAgentHandler` — GUI/browser interaction
- `OSWorldHandler` — OS simulation

### Runtime Layer (`openhands/runtime/`)
Multiple runtime backends: Docker, Singularity (HPC), Remote HTTP, Modal, Runloop. The Singularity runtime is the primary one used in this project for Slurm-based HPC environments. Uses Unix domain sockets (UDS) for runtime communication.

### Server (`openhands/server/`)
FastAPI + WebSocket server. The async evaluation server (`scripts/start_server.py`) exposes endpoints: `/start`, `/stop`, `/status`, `/add_llm_server`, `/clear_llm_server`, `/process`.

### CUA Module (`cua/`)
Custom data collection pipeline for multi-agent scenarios: goal generation (planner model), action generation (actor model), trajectory collection with VM-based execution, and post-processing/visualization tools.

### LLM Integration (`openhands/llm/`)
Uses LiteLLM for multi-provider LLM abstraction. Weighted round-robin load balancing across multiple LLM server endpoints.

## Key Configuration

- **Config file:** `config.toml` (created from `config.template.toml` or via `make setup-config`)
- **Environment:** Python 3.12+, Poetry for dependency management
- **Pre-commit config:** `dev_config/python/.pre-commit-config.yaml`
- **Dependency groups:** `dev` (ruff, mypy), `test` (pytest), `runtime` (jupyter), `evaluation` (swebench, datasets)

## Contributing

- All commits must be signed (`git commit -s`) per DCO requirements
- Run `make lint` before submitting PRs
- Follow existing conventions in the relevant module
- Keep PRs focused on a single concern
