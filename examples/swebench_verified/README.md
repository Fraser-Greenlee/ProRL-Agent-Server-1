# SWE-bench Verified Example

Evaluate Polar agent harnesses on the full [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) benchmark (500 human-validated tasks).

Each task runs an agent inside a per-instance container with the repo at `base_commit`, then grades the resulting patch via `swebench.harness.grading`.


## Installation

```bash
uv venv
uv pip install -e .
uv pip install --prerelease=allow sglang==0.5.10
bash scripts/patch/patch_sglang.sh
```

Install host-side evaluator dependencies (swebench grading + HuggingFace datasets):

```bash
bash examples/swebench_verified/setup_host.sh
```

## Quick Start

### 1. Start SGLang backends

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.7

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8001 \
    --mem-fraction-static 0.7
```

### 2. Start Polar services

```bash
uv run polar serve_rollout -c examples/swebench_verified/topology.yaml
uv run polar serve_gateway -c examples/swebench_verified/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/swebench_verified/topology.yaml --node-id localhost-node-02
```

### 3. Build runtime images

```bash
# Build all 500
uv run python examples/swebench_verified/build_images.py

# Or build a subset
uv run python examples/swebench_verified/build_images.py --max-tasks 10
```

### 4. Submit tasks

```bash
# Run all 500 tasks for pass@1
uv run python examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology examples/swebench_verified/topology.yaml \
  --runtime-backend docker \
  --num-samples 1 \
  --max-concurrent 4 \
  --max-tasks 10

# pass@8 for first 10 tasks
uv run python examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology examples/swebench_verified/topology.yaml \
  --runtime-backend docker \
  --num-samples 8 \
  --max-concurrent 4 \
  --max-tasks 10
```
