# SWE-Gym Slime GRPO (Hierarchical Advantage)

Fully async RL training on the curated 10-task SWE-Gym sample using **Polar** for agent rollout and **Slime** for distributed training with native GPU-to-GPU weight sync. Polar owns advantage estimation via its `HierarchicalGroupAdvantageEstimator`, which decomposes credit into between-trajectory (GRPO-style) and within-trajectory (per-trace) components.

## Architecture

```
                        ┌─────────────────────────────┐
                        │  Slime train_async.py        │
                        │  Megatron GRPO (GPU 4-7)     │
                        │  TP=2, DP=2                  │
                        └──────────┬──────────────────┘
                                   │  generate_rollout_polar_async()
                         weight sync (NCCL) every step
                                   │
┌──────────────────┐    ┌──────────┴───────────────────┐
│  SGLang ×4       │    │  Polar Rollout  :8080        │
│  GPU 0-3         │◀───│    └─ Gateway  :8100         │
│  (Slime-managed) │    │         └─ SWE-Agent in      │
│  Router :9000    │    │            Apptainer (CPU)    │
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
pip install -e .

# 2. Clone and install Slime (training framework) — pinned to v0.2.4
git clone --branch v0.2.4 --depth 1 git@github.com:THUDM/slime.git slime
pip install -e slime
# See slime/build_conda.sh for full dependency list
# (megatron-core, transformer_engine, flash_attn, apex, ray, etc.)

# 3. Pin SGLang to the patched version
pip install --prerelease=allow sglang==0.5.10

# 4. Clone Megatron-LM (needed for training internals)
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
pip install -e Megatron-LM

# 5. Build SWE-Agent container images
python examples/swegym_slime_grpo/build_images.py

# 6. Apply SGLang patch (adds token IDs to logprobs) — expects sglang==0.5.10
bash scripts/patch/patch_sglang.sh

# 7. Apply Slime patch (adds external advantage estimator support) — expects slime==0.2.4
bash scripts/patch/patch_slime.sh
```

## Quick Start

```bash
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

## Advantage Estimation

Polar computes per-trace advantages before handing samples to Slime (`--advantage-estimator external`). The `HierarchicalGroupAdvantageEstimator` decomposes the advantage for each trace into two levels:

```
A(trace j in traj i) = (1/K_i) * [A_between(i) + β · A_within(i,j)]
```

- **A_between** — GRPO-style: how good is this trajectory vs others in the group
- **A_within** — credit assignment: how good is this trace vs siblings in the same trajectory
- **1/K_i** — normalizes so each trajectory contributes equally regardless of trace count

All traces in a trajectory go to training — builders own trace curation (e.g. `all_records` for turn-by-turn, `prefix_merging` for aggregated chains). Configure in `polar_config.yaml` under `polar_adv_estimator`. Set `beta: 0` for pure GRPO behavior. Remove the block entirely to fall back to Slime's built-in estimators.

### Off-policy correction (`--use-tis`)

When `--advantage-estimator external` is combined with `--use-rollout-logprobs`, the trainer must also set `--use-tis` so samples that straddle a weight update receive truncated importance sampling correction. Without `--use-tis`, those samples train uncorrected — silently degrading signal. `run.sh` sets this flag; keep it when deriving new configs.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Main launch: starts Polar, Ray + Slime (manages SGLang), and training |
| `polar_config.yaml` | Polar bridge config (task template, concurrency, advantage estimator) |
| `topology.yaml` | Polar cluster topology (rollout + gateway, points at Slime's SGLang router) |
| `prepare_data.py` | Fetches SWE-Gym tasks, writes JSONL for Slime |
| `convert_weights.sh` | HF to Megatron torch_dist weight conversion |
