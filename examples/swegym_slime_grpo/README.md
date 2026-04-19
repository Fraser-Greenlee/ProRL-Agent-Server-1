# SWE-Gym Slime GRPO

Fully async RL training on the curated 10-task SWE-Gym sample using **Polar** for agent rollout and **Slime** for distributed training with native GPU-to-GPU weight sync. Advantage estimation lives entirely inside Slime: the rollout adapter tags every trace in a trajectory with the same `Sample.index`, and Slime's `--custom-reward-post-process-path` hook collapses them to one outcome reward per trajectory before GRPO normalizes across trajectories in the group. 

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

## Prerequisites

```bash
# 1. Install Polar
uv pip install -e .

# 2. Clone and install Slime (training framework) — pinned to v0.2.4
git clone --branch v0.2.4 --depth 1 git@github.com:THUDM/slime.git slime
uv pip install -e slime
# See slime/build_conda.sh for full dependency list
# (megatron-core, transformer_engine, flash_attn, apex, ray, etc.)

# 3. Pin SGLang to the patched version
uv pip install --prerelease=allow sglang==0.5.10

# 4. Clone Megatron-LM (needed for training internals)
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
uv pip install -e Megatron-LM

# 5. Build per-instance runtime container images
python examples/swegym_slime_grpo/build_images.py

# 6. Apply SGLang patch (adds token IDs to logprobs) — expects sglang==0.5.10
bash scripts/patch/patch_sglang.sh
```

## Quick Start

```bash-0
# Prepare training data (fetches 10 SWE-Gym tasks from HuggingFace)
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

## Files

| File | Purpose |
|---|---|
| `run.sh` | Main launch: starts Polar, Ray + Slime (manages SGLang), and training |
| `polar_config.yaml` | Polar bridge config (task template, concurrency) |
| `topology.yaml` | Polar cluster topology (rollout + gateway, points at Slime's SGLang router) |
| `prepare_data.py` | Fetches SWE-Gym tasks, writes JSONL for Slime |
| `convert_weights.sh` | HF to Megatron torch_dist weight conversion |
