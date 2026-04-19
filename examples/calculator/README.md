# Calculator Example

Simple "create a python calculator" rollout example using `polar`.

The example recommends 2 x H100 or comparable GPUs on a local machine:

- one rollout service on `:8080`
- two gateway nodes on `:8100` and `:8101`
- two local SGLang backends on `:8000` and `:8001`
- one topology file at [topology.yaml](topology.yaml)

## Installation

```bash
uv venv
uv pip install -e .
uv pip install --prerelease=allow sglang==0.5.10
bash scripts/patch/patch_sglang.sh
```

The patch supports TITO in sglang for OAI Chat Completion.

## Quick Start

### 1. Start SGLang backends

Start two SGLang servers, one per GPU group:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m sglang.launch_server \
   --model-path Qwen/Qwen3.5-4B \
   --host 0.0.0.0 \
   --port 8000 \
   --tp-size 2 \
   --tool-call-parser qwen3_coder \
   --reasoning-parser qwen3 \
   --mem-fraction-static 0.7 \
   --context-length 262144 \
   --trust-remote-code

CUDA_VISIBLE_DEVICES=2,3 uv run python -m sglang.launch_server \
   --model-path Qwen/Qwen3.5-4B \
   --host 0.0.0.0 \
   --port 8001 \
   --tp-size 2 \
   --tool-call-parser qwen3_coder \
   --reasoning-parser qwen3 \
   --mem-fraction-static 0.7 \
   --context-length 262144 \
   --trust-remote-code
```

### 2. Start Polar services

```bash
uv run polar serve_rollout -c examples/calculator/topology.yaml
uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/calculator/topology.yaml --node-id localhost-node-02
```

### 3. Build the shared runtime image

Build once for all harnesses:

```bash
uv run python examples/calculator/build_image.py
```

### 4. Submit tasks

For single harness:

```bash
uv run python examples/calculator/submit_calculator_task.py \
  --harness claude_code \
  --topology examples/calculator/topology.yaml \
  --runtime-backend docker \
  --num-samples 4
```

Or to submit on all harnesses:

```bash
uv run python examples/calculator/submit_all.py --num-samples 4
```

- `claude_code`
- `codex`
- `gemini_cli`
- `opencode`
- `pi`
- `qwen_code`

