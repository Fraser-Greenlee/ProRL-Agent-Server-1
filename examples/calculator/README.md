# Calculator Example

Simple "create a python calculator" rollout example using `polar`. 

The example recommends 2 x H100 or comparable GPUs on a local machine:

- one rollout service on `:8080`
- two gateway nodes on `:8100` and `:8101`
- two local SGLang backends on `:8000` and `:8001`,
- one topology file at [topology.yaml](topology.yaml)

## Installation

```bash
uv pip install -e .
uv pip install --upgrade sglang
source .venv/bin/activate && bash scripts/patch/patch_sglang.sh
```

The patch supports TITO in sglang for OAI Chat Completion.

## Quick Start

1. Start two SGLang servers (one per GPU) in separate terminals:

   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.7 --context-length 131072 --max-running-requests 2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder

   CUDA_VISIBLE_DEVICES=1 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8001 --tp-size 1 --mem-fraction-static 0.7 --context-length 131072 --max-running-requests 2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
   ```

   These conservative settings keep the calculator example stable on a single 4B model while preserving tool calling and training traces.

2. Start the services in three more terminals:

   ```bash
   uv run polar serve_rollout -c examples/calculator/topology.yaml
   uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-01
   uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-02
   ```
   
   Monitor with:

   ```bash
   watch -n 1 uv run polar status -c examples/calculator/topology.yaml
   ```


3. Build the harness image you want to test. For `claude_code`:

   ```bash
   bash examples/calculator/claude_code/setup.sh
   ```

4. Submit rollout:

   ```bash
   uv run python examples/calculator/claude_code/submit_tasks.py --num-samples 16
   ```

The helper writes `request.json`, submits it through `polar submit`, and stores `response.json` next to it.

## What The Task Does

Each session gets the same instruction: write `calculator.py`, expose a `Calculator` class, and pass the canonical test file in `assets/test_calculator.py`.

The runtime is prepared by:

- creating `/polar/session/workspace`
- initializing a fresh git repo
- uploading the test file
- grading the resulting patch with `swegym_git_diff`

## Harness Matrix

| Harness | CLI | API | Docker image |
|---------|-----|-----|-------------|
| `codex` | `codex` | OpenAI Responses | `polar-localhost-codex:latest` |
| `opencode` | `opencode` | OpenAI Chat | `polar-localhost-opencode:latest` |
| `claude_code` | `claude` | Anthropic Messages | `polar-localhost-claude_code:latest` |
| `gemini_cli` | `gemini` | Google Generative AI | `polar-localhost-gemini_cli:latest` |
| `qwen_code` | `qwen` | OpenAI Chat | `polar-localhost-qwen_code:latest` |
| `openhands_sdk` | OpenHands SDK | OpenAI Chat | `polar-localhost-openhands_sdk:latest` |
| `swe_agent` | `sweagent` | OpenAI Chat | `polar-localhost-swe_agent:latest` |

## Outputs

Each run lands in:

```text
examples/calculator/<harness>/batches/<timestamp>/
  request.json
  response.json
```
