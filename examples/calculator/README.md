# Calculator Example

End-to-end rollout example that validates every built-in agent harness on a
simple Python calculator task.

## Task

Each agent receives the same instruction:

> Write a Python calculator with no extra imports. Support arithmetic
> expressions over integers and parentheses. Save it as `calculator.py`.
>
> Expose a `Calculator` class that can be called with a string expression.
>
> Example:
> ```python
> from calculator import Calculator
> cal = Calculator()
> print(cal("4*3-3"))  # should print 9
> ```

A canonical test file (`shared/assets/test_calculator.py`) is uploaded into
every runtime via `runtime.prepare` and executed by the `swegym_git_diff`
evaluator with `refresh_runtime=true`.

## Topology

All examples share the same localhost topology:

| Component | Address |
|-----------|---------|
| Rollout server | `http://127.0.0.1:8080` |
| Gateway node 01 | `http://127.0.0.1:8100` → vLLM `:8000` |
| Gateway node 02 | `http://127.0.0.1:8101` → vLLM `:8001` |

Both vLLM servers host `MiniMaxAI/MiniMax-M2.5`.

Gateway and rollout configs live in `shared/gateway_server.yaml` and
`shared/rollout_server.yaml`.

## Prerequisites

1. **Two vLLM servers** on ports 8000 and 8001 serving MiniMax-M2.5:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve MiniMaxAI/MiniMax-M2.5 \
     --tensor-parallel-size 4 --tool-call-parser minimax_m2 \
     --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice \
     --trust-remote-code --port 8000

   CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve MiniMaxAI/MiniMax-M2.5 \
     --tensor-parallel-size 4 --tool-call-parser minimax_m2 \
     --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice \
     --trust-remote-code --port 8001
   ```

2. **Docker** available for building and running agent containers.

3. **ARP installed** in a Python environment with access to `src/`:

   ```bash
   export PYTHONPATH=/path/to/nv-arp/src
   ```

## Starting the servers

```bash
# Terminal 1 — rollout server
CONFIG_PATH=examples/calculator/shared/rollout_server.yaml \
  python -m rollout.server

# Terminal 2 — gateway node 01
CONFIG_PATH=examples/calculator/shared/gateway_server.yaml \
  GATEWAY_NODE_ID=localhost-node-01 \
  python -m gateway.server

# Terminal 3 — gateway node 02
CONFIG_PATH=examples/calculator/shared/gateway_server.yaml \
  GATEWAY_NODE_ID=localhost-node-02 \
  python -m gateway.server
```

## Running an agent

Each harness directory contains a `Dockerfile`, `setup.sh`, and
`submit_tasks.py`.

```bash
# 1. Build the Docker image
cd examples/calculator/opencode
bash setup.sh

# 2. Submit the task (from the repo root)
python examples/calculator/opencode/submit_tasks.py \
  --num-rollouts 16
```

The submit script delegates to `shared/submit_calculator_task.py`, which
constructs a `TaskRequest` with the correct harness name, container image,
`runtime.prepare` actions, and `swegym_git_diff` evaluator config.

Results are written to `<harness>/batches/<timestamp>/`.

## Harnesses

| Harness | CLI | API | Docker image |
|---------|-----|-----|-------------|
| `opencode` | `opencode` | OpenAI Chat | `arp-localhost-opencode:latest` |
| `claude_code` | `claude` | Anthropic Messages | `arp-localhost-claude_code:latest` |
| `codex` | `codex` | OpenAI Responses | `arp-localhost-codex:latest` |
| `gemini_cli` | `gemini` | Google GenerativeAI | `arp-localhost-gemini_cli:latest` |
| `qwen_code` | `qwen` | OpenAI Chat | `arp-localhost-qwen_code:latest` |
| `openhands_sdk` | OpenHands SDK | OpenAI Chat | `arp-localhost-openhands_sdk:latest` |
| `swe_agent` | `sweagent` | OpenAI Chat | `arp-localhost-swe_agent:latest` |

### Custom flags

Pass extra arguments through to the shared submit script:

```bash
python examples/calculator/opencode/submit_tasks.py \
  --num-rollouts 4 \
  --timeout-seconds 600 \
  --agent-timeout 300 \
  --model-name openai/MiniMaxAI/MiniMax-M2.5
```

## Evaluator

All examples use the `swegym_git_diff` evaluator in expected-output mode with
`refresh_runtime=true`:

1. The agent runtime's git diff is captured (`git add -A && git diff --cached`).
2. A fresh evaluator runtime replays `runtime.start() + runtime.prepare`.
3. The patch is applied to the fresh runtime.
4. `python3 test_calculator.py` runs the canonical test suite.
5. Reward is 1 if the output parser observes `PASSED test_calculator`, else 0.
