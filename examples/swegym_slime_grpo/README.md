# SWE-Gym Slime GRPO

Fully async RL training on 30 curated SWE-Gym sample using **Polar** for agent rollout and **Slime** for distributed training with native GPU-to-GPU weight sync.

Base model: **Qwen/Qwen3.5-4B** (VLM checkpoint, trained text-only; hybrid attention
with 1 full + 3 GatedDeltaNet linear per 4 layers).

This demo runs on single node (8 x B200).

## Architecture

```
                        ┌─────────────────────────────┐
                        │  Slime train_async.py       │
                        │  Megatron GRPO (GPU 4-7)    │
                        │  TP=2, DP=2                 │
                        └──────────┬──────────────────┘
                                   │  generate_rollout_polar_async()
                         weight sync (NCCL) every step
                                   │
┌──────────────────┐    ┌──────────┴───────────────────┐
│  SGLang ×4       │    │  Polar Rollout  :8080        │
│  GPU 0-3         │◀───│    └─ Gateway  :8100         │
│  (Slime-managed) │    │         └─ Agent harness in  │
│  Router :9000    │    │            Apptainer (CPU)   │
└──────────────────┘    └──────────────────────────────┘
```

| GPU   | Role                                        | Port       |
|-------|---------------------------------------------|------------|
| 0-3   | SGLang inference (4 engines, Slime-managed, weight-synced) | 9000 (router) |
| 4-7   | Megatron GRPO training (Ray)                | 8265 (Ray) |
| CPU   | Polar rollout + gateway node                | 8080, 8100 |

## Installation

```bash
# 1. Install Polar
uv pip install -e .

# 2. Clone and install Slime (training framework) — pinned to v0.2.4
git clone --branch v0.2.4 --depth 1 git@github.com:THUDM/slime.git slime
uv pip install -e slime
# See slime/build_conda.sh for full dependency list
# (megatron-core, transformer_engine, flash_attn, apex, ray, etc.)

# 3. Qwen3.5 GatedDeltaNet needs flash-linear-attention
uv pip install --prerelease=allow flash-linear-attention

# 4. Install flash-attn 2.7.4.post1 from source (B200 / SM100).
#    Qwen3.5-4B uses head_dim=256. Transformer Engine 2.5.0's bundled cuDNN
#    backend does NOT support (head_dim=256, thd layout) on SM100, so slime
#    falls back to FlashAttention — which TE 2.5.0 only recognizes at the
#    flash-attn 2.x API (the 4.x package coexists but is not used here).
TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS=32 NVCC_THREADS=4 \
    uv pip install --no-build-isolation flash-attn==2.7.4.post1

# 5. Pin SGLang to the patched version
uv pip install --prerelease=allow sglang==0.5.10

# 6. Clone Megatron-LM (needed for training internals)
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
uv pip install -e Megatron-LM

# 7. Build per-instance runtime container images
python examples/swegym_slime_grpo/build_images.py

# 8. Apply SGLang patches:
#    - token-ids-in-logprobs patch (expects sglang==0.5.10)
#    - VLM text-only input_ids patch (Qwen3.5-4B is a VLM checkpoint served
#      text-only; SGLang otherwise drops input_token_ids for text chat)
bash scripts/patch/patch_sglang.sh
```

## Quick Start

```bash-0
# Prepare training data (fetches 30 SWE-Gym tasks from HuggingFace)
python examples/swegym_slime_grpo/prepare_data.py

# Convert HF weights to Megatron format
bash examples/swegym_slime_grpo/convert_weights.sh

# Run everything (Polar, Ray + Slime + SGLang, training)
bash examples/swegym_slime_grpo/run.sh
```

Override paths if cloned elsewhere:
```bash
SLIME_DIR=/path/to/slime MEGATRON_DIR=/path/to/Megatron-LM bash run.sh
```

### Advantage estimation

`run.sh` runs Slime's built-in GRPO (`--advantage-estimator grpo --grpo-std-normalization`) with `--custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards`. The hook dedupes traces from the same Polar session to one reward per trajectory, then normalizes across trajectories in the group — the shared-within-trajectory, normalized-across-trajectories behavior previously implemented as a Polar-side estimator now lives in a 60-line Slime plugin.

### Off-policy correction (`--use-tis`)

When `--use-rollout-logprobs` is set, the trainer must also set `--use-tis` so samples that straddle a weight update receive truncated importance sampling correction. Without `--use-tis`, those samples train uncorrected — silently degrading signal. `run.sh` sets this flag; keep it when deriving new configs.

### Qwen3.5-4B specifics

The checkpoint is a VLM (`Qwen3_5ForConditionalGeneration`) trained text-only here.
A few non-obvious requirements follow from that:

- **List-format prompts**. When the HF checkpoint ships a processor, slime asserts the dataset prompt is a chat message list (`slime/slime/utils/data.py:243`). `prepare_data.py` already emits `[{"role": "user", "content": ...}]`; stick to that shape if you fork the dataset builder.
- **flash-attn 2.x required**. head_dim=256 + packed (`thd`) varlen has no cuDNN backend on SM100 in TE 2.5.0, so training routes to FlashAttention. Install flash-attn 2.7.4.post1 (step 4 above) — the flash-attn 4.x prerelease is not recognized by TE 2.5.0.
- **Tool-call parser `qwen3_coder`**. Qwen3.5-4B emits Qwen3-Coder–style XML: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`. The `qwen25` parser tries to JSON-parse the XML body and silently produces zero tool calls (SGLang logs `</function>, JSON parse error: Expecting value: line 1 column 1`). `run.sh` uses `--sglang-tool-call-parser qwen3_coder`; changing it will collapse rewards to 0.
- **qwen-code harness over opencode**. opencode's ~10K-token tool-schema system prompt silences 4B-class Qwen models; qwen-code is Qwen's own gemini-cli fork with a much leaner prompt and maps cleanly to the OpenAI-compat shape SGLang serves.
- **`--max-tokens-per-gpu 100000`**. Qwen3.5-4B actor_train with TP=2, DP=2 peaks at ~181GB per B200 (of 183GB) under `--use-dynamic-batch-size`. The slime default (200000) OOMs on this checkpoint — the hybrid GatedDeltaNet + full-attention layer layout has higher activation memory than pure dense Qwen3. 100000 gives 2GB headroom and keeps dynamic batching engaged.

### Concurrency & worker sizing

Two knobs control session fan-out, and they **must** be kept in sync:

| Knob | File | Meaning |
|---|---|---|
| `polar_max_concurrency` | `polar_config.yaml` | Slime-side cap: max groups the trainer dispatches in parallel |
| `max_{init,run,postrun}_workers` | `topology.yaml` | Node-side cap: max sessions actually running in parallel |

If `polar_max_concurrency` > `max_run_workers`, excess sessions queue at the node and eventually time out (`timeout_seconds`), returning as empty placeholders with reward 0. The `reward_post_process` hook zeros their advantages so training continues as a no-op, but the step is wasted.

This repo ships with **32 / 32 / 32 / 32** — aligned. Prior 64-wide runs saturated the
Polar gateway's httpx connection pool (thousands of `httpx.PoolTimeout` errors as sessions
multiplied LLM-call + polling traffic), which surfaced as slowly-climbing empty-rollout
counts. 32 workers fits well within the pool and keeps all eight B200s busy — the host
(224 cores / 2 TB RAM) is nowhere near CPU/RAM-bound at this size. On a smaller machine,
lower all four together. `timeout_seconds: 5400` gives long-running SWE-Gym sessions
(test runs, large repos) room to finish.

### Observed behavior on 8×B200

Validated 2026-04-21 on this host (8×B200 / SM100, TE 2.5.0, flash-attn 2.7.4.post1, sglang 0.5.10). Async overlap is fully engaged after the first rollout: `train_wait_time` drops from ~26 min (cold start, no rollout available yet) to ~2-5 s for the rest of the run. NCCL GPU-to-GPU weight sync completes in <2 s for 4B params.

Reward signal is noisy within a single run (4-prompt batches × 8 trajectories) but peaks >0.9 across the first few rollouts. Steady-state `reward_std` sits ~0.35-0.45 — healthy GRPO variance. Expect `rollout/truncated_ratio = 0.0` throughout (prompts are short, responses well within 8 K).

Memory headroom is tight: actor_train peaks at ~180 GB of 183 GB per training GPU under `--use-dynamic-batch-size --max-tokens-per-gpu 100000`. Do NOT raise `--max-tokens-per-gpu` on this checkpoint.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Main launch: starts Polar, Ray + Slime (manages SGLang), and training |
| `polar_config.yaml` | Polar bridge config (task template, concurrency) |
| `topology.yaml` | Polar cluster topology (rollout + gateway, points at Slime's SGLang router) |
| `prepare_data.py` | Fetches SWE-Gym tasks, writes JSONL for Slime |
| `convert_weights.sh` | HF to Megatron torch_dist weight conversion |
