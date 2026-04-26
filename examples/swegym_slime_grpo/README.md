# SWE-Gym Slime GRPO

Fully async RL training on the full SWE-Gym SkyRL split using **Polar** for agent rollout and **Slime** for distributed training with native GPU-to-GPU weight sync.

Base model: **Qwen/Qwen3.5-4B** (VLM checkpoint, trained text-only; hybrid attention
with 1 full + 3 GatedDeltaNet linear per 4 layers).

This demo runs on single node (8 x B200).

## Architecture

```
                        ┌─────────────────────────────┐
                        │  Slime train_async.py       │
                        │  Megatron GRPO (GPU 0-3)    │
                        │  TP=2, DP=2                 │
                        └──────────┬──────────────────┘
                                   │  generate_rollout_polar_async()
                         weight sync (NCCL) every step
                                   │
┌──────────────────┐    ┌──────────┴───────────────────┐
│  SGLang ×4       │    │  Polar Rollout  :8080        │
│  GPU 4-7         │◀───│    └─ Gateway  :8100         │
│  (Slime-managed) │    │         └─ Agent harness in  │
│  Router :9000    │    │            Docker (CPU)      │
└──────────────────┘    └──────────────────────────────┘
```

| GPU   | Role                                        | Port       |
|-------|---------------------------------------------|------------|
| 0-3   | Megatron GRPO training (Ray, TP=2, DP=2)    | 8265 (Ray) |
| 4-7   | SGLang inference (4 engines, Slime-managed, weight-synced) | 9000 (router) |
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

# 7. Prepare the shared agent CLI directory and pull train/validation base images
python examples/swegym_slime_grpo/build_images.py

# 8. Apply SGLang patches:
#    - token-ids-in-logprobs patch (expects sglang==0.5.10)
#    - VLM text-only input_ids patch (Qwen3.5-4B is a VLM checkpoint served
#      text-only; SGLang otherwise drops input_token_ids for text chat)
bash scripts/patch/patch_sglang.sh
```

## Quick Start

```bash
# Prepare checkouts/data/images/checkpoint as needed, then run everything
bash examples/swegym_slime_grpo/launch_e2e.sh
```

`run.sh` trains for one epoch over `swegym_train_293.jsonl` and evaluates
`swegym_eval_23.jsonl` every 8 rollouts and at the epoch boundary. Slime uses
fixed-size rollout batches, so the example uses
`slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer`
to run 37 train rollouts: all 293 train prompts are consumed once, and the final
batch wraps 3 prompts to fill the batch. This gives regular validation reward
points for reporting without launching eval after every trainer update.

Override paths if cloned elsewhere:
```bash
SLIME_DIR=/path/to/slime MEGATRON_DIR=/path/to/Megatron-LM \
  bash examples/swegym_slime_grpo/launch_e2e.sh
```

### GPU utilization monitoring

W&B's built-in System GPU panel can show dozens of process-scoped labels because
Ray, Slime, and secondary W&B processes all attach to the shared run. For a
stable per-GPU and per-role view, attach the sidecar monitor after the W&B run id
is known:

```bash
uv run python scripts/monitor_wandb_gpu.py \
  --wandb-run-id <run-id> \
  --wandb-project polar-swegym-grpo \
  --train-gpus 0,1,2,3 \
  --rollout-gpus 4,5,6,7 \
  --out-csv tmp/gpu_monitor/<run-id>_gpu.csv
```

The monitor logs stable W&B keys such as
`polar_system/gpu_rollout/mean_util_pct`,
`polar_system/gpu_rollout/min_util_pct`,
`polar_system/gpu_train/mean_util_pct`, and
`polar_system/gpu_2/util_pct`. Use these for throughput plots instead of the
auto-generated `System/GPU utilization` labels.

### Slime compatibility patch

`run.sh` applies `scripts/patch/slime_polar_async.patch` to the external Slime checkout before launching training. The patch is intentionally small: it lets Slime notify the Polar rollout bridge after serving weights advance and carries an experimental pre/post hook for gateway-paused weight sync.

If Slime has already been patched, the script skips it. If the patch no longer applies, stop and port the patch to the new Slime revision before running this example.

### Advantage estimation

`run.sh` runs Slime's built-in GRPO (`--advantage-estimator grpo --grpo-std-normalization`) with `--custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards`. The hook dedupes traces from the same Polar session to one reward per trajectory, then normalizes across trajectories in the group — the shared-within-trajectory, normalized-across-trajectories behavior previously implemented as a Polar-side estimator now lives in a 60-line Slime plugin.

### Off-policy correction (`--use-tis`)

Polar supplies real behavior-policy `rollout_log_probs` on every trainable response token. `run.sh` enables `--use-tis` so samples collected by a stale served policy receive truncated importance sampling correction. Do **not** add `--use-rollout-logprobs` here: Slime treats that as a separate mode and asserts it cannot be combined with `--use-tis`.

The Polar bridge bounds async drift with `polar_max_async_level` and `polar_max_off_policy_steps`. Completed groups that exceed the staleness bound are dropped before training, and failed, empty, or zero-trainable-token groups are also dropped so one bad Polar task does not stop the run. Each task is submitted once.

### Qwen3.5-4B specifics

The checkpoint is a VLM (`Qwen3_5ForConditionalGeneration`) trained text-only here.
A few non-obvious requirements follow from that:

- **List-format prompts**. When the HF checkpoint ships a processor, slime asserts the dataset prompt is a chat message list (`slime/slime/utils/data.py:243`). `prepare_data.py` already emits `[{"role": "user", "content": ...}]`; stick to that shape if you fork the dataset builder.
- **flash-attn 2.x required**. head_dim=256 + packed (`thd`) varlen has no cuDNN backend on SM100 in TE 2.5.0, so training routes to FlashAttention. Install flash-attn 2.7.4.post1 (step 4 above) — the flash-attn 4.x prerelease is not recognized by TE 2.5.0.
- **Tool-call parser `qwen3_coder`**. Qwen3.5-4B emits Qwen3-Coder–style XML: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`. The `qwen25` parser tries to JSON-parse the XML body and silently produces zero tool calls (SGLang logs `</function>, JSON parse error: Expecting value: line 1 column 1`). `run.sh` uses `--sglang-tool-call-parser qwen3_coder`; changing it will collapse rewards to 0.
- **Router policy `round_robin`**. SGLang's default `cache_aware` policy can over-stick SWE-Gym traffic to the engines with the hottest shared prefixes. This example defaults to `--router-policy round_robin` so all four rollout engines carry load during throughput tests. Override `SGLANG_ROUTER_POLICY=cache_aware` only when prefix-cache hit rate matters more than balanced GPU utilization.
- **Codex harness**. The example defaults to `codex` so Qwen3.5 is evaluated through a harness it is less likely to have overfit during pretraining or instruction tuning. Codex talks the OpenAI Responses API through the Polar gateway; the gateway transforms requests to the Slime-managed SGLang router.
- **`--max-tokens-per-gpu 80000`**. Qwen3.5-4B actor_train with TP=2 peaks close to the B200 memory limit under `--use-dynamic-batch-size`. The slime default (200000) OOMs on this checkpoint — the hybrid GatedDeltaNet + full-attention layer layout has higher activation memory than pure dense Qwen3. 80000 keeps dynamic batching engaged while leaving enough headroom for long SWE-Gym traces.

### Concurrency & worker sizing

These knobs control session fan-out and async backlog:

| Knob | File | Meaning |
|---|---|---|
| `polar_max_concurrency` | `polar_config.yaml` | Slime-side cap: max groups the trainer dispatches in parallel |
| `polar_max_session_concurrency` | `polar_config.yaml` | Slime-side cap: max Polar sessions in flight across groups |
| `polar_max_async_level` | `polar_config.yaml` | Max accepted/active rollout batches ahead of training |
| `polar_max_off_policy_steps` | `polar_config.yaml` | Max rollout/train policy-version gap accepted for training |
| `polar_allow_weight_update_overlap` | `polar_config.yaml` | Experimental. Keep `false` for this example; SGLang weight sync can hang if forced while rollout is active |
| `max_{init,run,postrun}_workers` | `topology.yaml` | Node-side cap: max sessions actually running in parallel |

`polar_max_session_concurrency` should cover the sessions admitted by Slime-side
prompt-group concurrency. Failed, empty, or too-stale groups are dropped by the
bridge scheduler before training rather than padded into the optimizer step.
For this safe-sync setup, keep
`polar_max_off_policy_steps >= polar_max_async_level + update_weights_interval`:
Slime waits for the next generated batch before syncing weights, so a
two-batch async backlog with per-rollout weight sync naturally produces
samples up to three policy steps stale.

This repo ships with **8 prompt groups per rollout**, **128 Slime-side session slots**,
and gateway workers sized as **32 init / 32 run / 32 postrun**.
A prior higher-init run saturated the
docker daemon (many sessions x 2 containers, each running repeated Node/agent CLI setup at
session start), producing
`docker create failed with exit code -1`, `git diff ... exit code 137` (SIGKILL'd
mid-exec), and `docker rm -f failed` cascades. Two changes fix both the daemon pressure
and the wasted work: (1) Node 22 + agent CLIs are prepared once under
`tmp/swegym_agent_cli/opt_node` and mounted read-only into the official SWE-Gym
base images, so `prepare` is just workspace copy + symlinks; (2) a separate
`eval_prepare` runs in the eval container without agent setup work.
Most SkyRL rows use `xingyaoww/sweb.eval.x86_64.{owner}_s_{repo}-{id}` images;
a small set of legacy SWE-bench repos use
`swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{id}` instead. `sample_tasks.py`
selects the right namespace when generating JSONL rows.
The 32 / 32 / 32 gateway-worker split targets the four rollout engines on this host
(224 cores / 2 TB RAM) while throttling Docker create/remove pressure. Trainer GPU
utilization still depends on how quickly Polar can
produce complete accepted groups; long-running SWE-Gym sessions can leave
Megatron waiting even while SGLang GPUs are saturated. On a smaller machine
lower prompt-group, session, and gateway-worker limits together. This example uses
`timeout_seconds: 1200` to cap long-tail sessions before they hold batch completion
too long.

### Observed behavior on 8×B200

Validated 2026-04-24 on this host (8×B200 / SM100, TE 2.5.0, flash-attn 2.7.4.post1, sglang 0.5.10). Directly calling SGLang weight sync while rollout is active can hang the update, even when the Polar gateway has paused new LLM calls and reports zero in-flight SGLang requests. The example therefore keeps `polar_allow_weight_update_overlap: false`: Slime finishes the next generated batch before syncing weights, while Polar still uses bounded buffering, dropped-group accounting, fully masked placeholders, and policy-version tagging.

Reward signal is noisy within a single run (8-prompt batches x 8 trajectories)
and is harness-sensitive. Use the repeated 23-task eval points
for reporting, and watch that `reward_std` stays non-zero so GRPO has within-group
learning signal. Expect `rollout/truncated_ratio = 0.0` throughout (prompts are
short, responses well within 8 K).

Memory headroom is tight: actor_train can run close to the 183 GB B200 limit under `--use-dynamic-batch-size --max-tokens-per-gpu 80000`. Do NOT raise `--max-tokens-per-gpu` on this checkpoint without revalidating memory. The current example gives four GPUs to Megatron and four GPUs to rollout; override `ACTOR_NUM_GPUS_PER_NODE` and `ROLLOUT_NUM_GPUS` only after checking train memory and rollout utilization.

## Files

| File | Purpose |
|---|---|
| `launch_e2e.sh` | Single-entry E2E launcher: prepares dependencies/assets, then calls `run.sh` |
| `run.sh` | Main launch: starts Polar, Ray + Slime (manages SGLang), and training |
| `polar_config.yaml` | Polar bridge config (task template, concurrency) |
| `topology.yaml` | Polar cluster topology (rollout + gateway, points at Slime's SGLang router) |
| `prepare_data.py` | Fetches the full SWE-Gym SkyRL train/validation split, writes JSONL for Slime |
| `convert_weights.sh` | HF to Megatron torch_dist weight conversion |
