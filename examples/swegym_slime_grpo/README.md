# SWE-Gym Slime GRPO

End-to-end example: train **Qwen3.5-4B** with async **GRPO** on **SWE-Gym** tasks,
using **Polar** for agent rollouts and **Slime** for training. Targets a single
node with 8× B200.

## Quick Start

```bash
bash examples/swegym_slime_grpo/launch_e2e.sh
```

`launch_e2e.sh` is the one-shot entry point: it clones Slime + Megatron-LM,
applies the Slime/SGLang patches, builds the 293-task SWE-Gym JSONL, pulls
Apptainer images and the shared agent CLIs, converts the Qwen weights to
torch_dist, then hands off to `run.sh`.

For Slime/Megatron install details see
[../../src/slime_bridge/README.md](../../src/slime_bridge/README.md#slime-installation).


## Files

| File | Purpose |
|---|---|
| `launch_e2e.sh` | One-shot entry: setup + run |
| `run.sh` | Launches Polar services + Ray + Slime training job |
| `convert_weights.sh` | HF checkpoint → Megatron torch_dist |
| `model_args.sh` | Qwen3.5-4B Megatron args, shared by `run.sh` + `convert_weights.sh` |
| `topology.yaml` | Polar topology template (`${SGLANG_ROUTER_BASE_URL}` filled at runtime) |
| `polar_config.yaml` | Polar bridge config template (`${AGENT_CLI_DIR}`, `${APPTAINER_IMAGE_DIR}` filled at runtime) |
| `prepare_data.py` | Builds `swegym_train_293.jsonl` |
| `prepare_apptainer_images.py` | Pulls per-task SIF images, builds shared Node + agent CLI dir |
| `sample_tasks.py` | Dataset helpers (HF fetch, registry image lookup) |

## Common knobs

| What you want to tune | Where |
|---|---|
| Train/rollout GPU split, batch size, KL coef, LR | `run.sh` (env vars near top + Slime args at bottom) |
| Which agent harness (qwen_code / claude_code / codex / opencode / pi) | `polar_config.yaml` → `agent.harness` |
| Per-task timeout, async level, callback host | `polar_config.yaml` → `polar_*` keys |
| Gateway/rollout host & port, model served | `topology.yaml` |
| Which SWE-Gym dataset / split | `sample_tasks.py` → `DATASET_NAME`, `DATASET_SPLITS` |
| Model architecture args (don't change unless swapping models) | `model_args.sh` |


