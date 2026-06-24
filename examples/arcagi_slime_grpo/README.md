# ARC-AGI Compression — Slime GRPO

End-to-end **training** example: train **Fraser/Qwen3.6-27B-ARC-Hy** with async
**GRPO** to compress ARC-AGI tasks into structural Hy programs, using **Polar**
for agent rollouts and **Slime** for training.

Each rollout drops the agent (OpenHands SDK harness) into a fresh container with
the auto-compress repo, points its LLM at the trained model, and asks it to
shrink one task's `tasks/arcagi/train/<id>.hy` while keeping every input/output
pair correct. The reward is computed by a custom evaluator:

> **reward = −1** if the final program isn't fully correct (or doesn't apply);
> otherwise **(baseline_size − new_size) / baseline_size**, clipped to **[0, 1]**.

So every incorrect rollout ranks below every correct one, and correct rollouts
form a continuum on how much they compress — exactly the signal GRPO needs.

> Status: this example trains on the **same 40 tasks** as the baseline eval run
> (first 40 of `compression-difficulty.tsv`, easy → hard). Every rollout starts
> from the **baseline scaffold** (raw-grid literals), so the model compresses
> from scratch each iteration — the point is to watch whether RL moves the
> reward at all. Curriculum / warm-starting from prior compressions is a later
> step (see *Future work*).

## Prerequisites

Install Polar and SGLang per the [top-level README](../../README.md#installation),
plus the Slime training stack (Slime + Megatron). The SWE-Gym example's
`launch_e2e.sh` automates the Slime/Megatron checkout and patches; reuse that
machinery — the only training-stack difference here is the model.

This example assumes the Convergence HPC layout:
- a sibling `auto-compress` checkout at `/home/fraser_convergence_ai/auto-compress`
- shared NFS at `/home` (survives node autoscaling)
- 8×H100 nodes on the `dev` partition; docker on workers; **no apptainer**

## Pipeline

### 1. Build the runtime image → save to NFS

The cluster's worker nodes are **autoscaled cloud VMs**: their `/var/lib/docker`
is wiped on power-down, so an image built in one job won't exist in the next. We
therefore `docker save` the image to a gzip tarball on the shared NFS and
`docker load` it at the start of each job.

```bash
# Builds polar-arcagi:latest on a CPU worker and saves it to
# /home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz
./run_remote.sh build-arcagi-image
```

The image bakes: the auto-compress repo at `/opt/auto-compress` with **every
task file reset to its baseline scaffold** (`eval.py --scaffold` — so no solved
sibling can leak answers), an arcagi venv (`/opt/arcagi-venv`: hy, hyrule,
arckit, scipy, numpy), and a baked OpenHands SDK 1.17 venv
(`/opt/openhands-sdk-venv`) so rollouts don't pip-install every session.

At the start of any rollout/training job, load it onto the local node:

```bash
bash examples/arcagi_slime_grpo/load_image.sh   # idempotent; no-op if present
```

### 2. Build the train JSONL

```bash
AUTO_COMPRESS=/home/fraser_convergence_ai/auto-compress \
  python examples/arcagi_slime_grpo/prepare_data.py --n-tasks 40
```

Writes `arcagi_train.jsonl` (40 rows). Each row: a chat `prompt` (the full
`baseline_run/prompt.py` instruction — task grids + LARC sidecar + arc-dsl
reference solver + baseline body + the whole `library.hy`), and `metadata` with
`task_id`, `baseline_size`, `baseline_body`. The reward evaluator reads
`baseline_size` from here. (Regenerated, not committed — it embeds library.hy
per row, ~2 MB.)

### 3. One-shot: bootstrap + convert + train

`launch_e2e.sh` does the whole bootstrap idempotently — clones Slime + Megatron,
applies the router-token + SGLang-0.5.13 patches, installs the editable packages
and training stack (Transformer Engine + Flash Linear Attention), builds the
rollout image to the NFS tarball (if missing), prepares the JSONL, converts the
HF checkpoint to Megatron torch_dist (if missing), then hands off to `run.sh`.

It **must run on a GPU node** (TE builds from source; conversion needs CUDA) and
holds the allocation for the whole run. Submit it via the slurm wrapper:

```bash
./run_remote.sh submit arcagi          # → sbatch job_arcagi_train.slurm (8×H100, dev)
./run_remote.sh logs                   # tail the latest job log
```

Re-submitting resumes: every bootstrap stage skips work already done.

`convert_weights.sh` (HF → Megatron torch_dist) and `run.sh` (Polar services +
Ray + Slime) mirror the SWE-Gym example; `model_args.sh` carries the Qwen3.6-27B
Megatron dimensions; GPU split is 4 train (TP=4) + 4 serve (TP=4), conservative
batch/recompute knobs (see *Open: parallelism* — values are estimates, the first
run may need a memory-tuning cycle). `--num-epoch 50` loops the 40-task bank
many times so we can watch the reward move.

### Files unique to the launch path

| File | Purpose |
|---|---|
| `launch_e2e.sh` | One-shot bootstrap + train (GPU node). |
| `convert_weights.sh` | HF → Megatron torch_dist (run once; `launch_e2e` calls it). |
| `run.sh` | Polar services + Ray + Slime `train_async.py`. |
| `../../job_arcagi_train.slurm` | Slurm wrapper: 8×H100 on `dev`, calls `launch_e2e.sh`. |

## Files

| File | Purpose |
|---|---|
| `arc_compress_evaluator.py` | Custom Polar evaluator: −1 / [0,1] compression reward. Subclasses `BasePatchEvaluator` (grades the agent's diff on a fresh clean runtime). |
| `prepare_data.py` | Builds `arcagi_train.jsonl` from the first N difficulty-ordered tasks. |
| `runtime/Dockerfile` | The rollout runtime image (auto-compress + venvs). |
| `build_image.py` | Builds the image and `docker save`s it to the NFS tarball. |
| `load_image.sh` | `docker load` the tarball on a fresh node (idempotent). |
| `polar_config.yaml` | Harness (openhands_sdk), runtime prepare/eval_prepare, evaluator wiring. |
| `topology.yaml` | Polar rollout/gateway topology; SGLang router URL filled at runtime. |
| `model_args.sh` | Qwen3.6-27B Megatron args (64 layers, hidden 5120, untied embeddings). |

## How the reward stays honest

The evaluator subclasses `BasePatchEvaluator`, so it extracts the agent's git
diff, **drops everything except the task file** (`exclude_patterns` covers
`library.hy`, `eval.py`, `reference/**`, …), applies just that onto a **fresh**
clean container, then runs `eval.py --task` / `--size` there. Edits to the
grader, the library, or sibling tasks never reach scoring. (We're not hardening
against determined reward-hacking yet — this is the baseline level of hygiene.)

## Open: parallelism (needs a decision before first run)

Qwen3.6-27B (BF16 ≈ 54 GB weights; Adam training state pushes well past one
80 GB H100) needs more sharding than the 4 B SWE-Gym example's `TP=2`. On one
8×H100 node the natural split is 4 train / 4 serve, but the train side likely
needs `TP=4` (+ activation recompute) and the serve side `TP=4` to hold 27B at
the rollout context length. `run.sh` is templated from SWE-Gym; the GPU split,
`--tensor-model-parallel-size`, `--rollout-num-gpus*`, batch sizes, and
`--max-tokens-per-gpu` still need tuning/validation for 27B.

## Training-stack setup gotchas (all handled by launch_e2e.sh)

These are the cluster-specific fixes that make the slime/Megatron stack actually
import and run here — each was a hard failure the first time. `launch_e2e.sh`
now does all of them; listed so they're not a mystery if something drifts.

- **Megatron commit, not tag.** slime v0.3.0 pins Megatron-LM commit
  `1dcf0dafa884ad52ffb243625717a3471643e087` (from its `docker/Dockerfile`).
  Tags like `26.04-alpha.rc1` dropped `megatron/training/tokenizer/`, which
  slime imports → `ModuleNotFoundError`.
- **slime's Megatron patch.** slime ships `docker/patch/latest/megatron.patch`
  (applied `git apply --3way`) that adds `--use-gated-attention`, the
  `use_gated_attention` config field + gate logic, and checkpoint-load fixes.
  Without it, conversion/training reject `--use-gated-attention`.
- **Transformer Engine 2.10.0** (matched to that Megatron commit; 2.5.0 ≠).
  Its `transformer-engine-torch` C++ build `#include`s `nccl.h` from the pip
  `nvidia-nccl` package, so that include/lib dir must be on `CPATH`/`LIBRARY_PATH`.
  And TE 2.x crashes at import doing `Path(nvidia.__file__=None)` unless
  `NVTE_CUDA_INCLUDE_DIR` is set.
- **numpy < 2.** Megatron hard-asserts `numpy<2`; the editable installs pull 2.x,
  so we force-reinstall `numpy<2` in the training venv (the rollout container has
  its own numpy, unaffected).
- **PATH on GPU nodes.** Non-login shells lack `uv` (`~/.local/bin`) and `nvcc`
  (`$CUDA_HOME/bin`); both are exported before any install/run step.

## Future work

- **Warm-start curriculum**: instead of resetting to the baseline scaffold every
  rollout, persist each task's best-so-far program and start subsequent rollouts
  from it, so compression accumulates across training (the user's intended
  long-run behavior).
- Allowlist-based diff filtering (only `tasks/arcagi/train/<id>.hy`) once we care
  about reward-hacking.
