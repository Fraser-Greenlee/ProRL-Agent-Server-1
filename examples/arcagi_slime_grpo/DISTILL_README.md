# Teacher distillation: Claude (OpenHands) → SFT data for Qwen3.6-27B

`distill_generate.py` runs **Claude via OpenHands** on the synthetic ARC compression
tasks and captures full trajectories (chain-of-thought + tool calls + tool results),
graded by the real `score_synthetic.py --verify`. Harvest the fully-correct ones into
SFT data and fine-tune the student. This sidesteps cold-start GRPO's wall: the weak
base policy leaves ~59% of tasks all-fail (zero-gradient groups), so it barely learns;
a strong teacher reliably produces solved, learnable trajectories on demand.

## Why this runs LOCALLY (not on the cluster)
The SFDC LLM Gateway resolves to a Salesforce-internal VPC ingress that the
Convergence HPC cluster **cannot route to** (curl times out). Fraser's laptop / SFDC
network can. So generation runs here; only the downstream SFT training runs on the
cluster.

## One-time setup
1. **Grader venv** (hy + the contract deps `score_synthetic.py` imports):
   ```
   uv venv /tmp/arcagi-grader --python 3.12
   uv pip install --python /tmp/arcagi-grader/bin/python \
       "hy>=1.0.0" "hyrule>=0.7" "numpy>=2.0" "scipy>=1.13" "arckit>=1.0.1"
   ```
2. **auto-compress** checkout present locally (default `~/Projects/auto-compress`).
3. **Gateway key** from the SFDC devbar → `SFDC_GATEWAY_KEY`. The SFDC CA bundle at
   `~/.aisuite/conf/npm-sfdc-certs.pem` must exist (the script loads it for TLS).

## Run (orchestrator, concurrent + resumable)
```
SFDC_GATEWAY_KEY=sk-... \
AUTO_COMPRESS=~/Projects/auto-compress \
ARCAGI_GRADER_PY=/tmp/arcagi-grader/bin/python \
OPENHANDS_SUPPRESS_BANNER=1 \
uv run --with openhands-sdk --with openhands-tools --with litellm --with certifi \
  python examples/arcagi_slime_grpo/distill_generate.py \
  --out runs/distill1 --tasks 1015 --refine
```
- Defaults: **Opus** (`anthropic/claude-opus-4-8`), 50 turns, **5 parallel** workers,
  extended thinking. Swap `--model anthropic/claude-sonnet-4-6` for cheaper/faster.
- `--refine`: after a task is solved, a SECOND independent trajectory is seeded with
  that solution to chase compression (where compression wins reliably come from).
- **Resumable**: one attempt per task; re-running skips tasks already attempted
  (`--force` to redo). Combined `trajectories.jsonl` is appended.

## Rate limit (the binding constraint)
The gateway allows **50 requests/min per key** (shared across workers & models). An
agent task makes ~20 calls. Each worker is a serial loop, so aggregate req/min ≈
`parallel × 60/call_seconds`; at ≥~6s/call (Opus+thinking), 5 workers stay well under
50/min. Patient 429-retry (num_retries=4, retry_max_wait=70 > the 60s reset) absorbs
bursts. A full 1015-task run takes hours — pace it, don't co-run other gateway load.

## Robustness
Each task runs as a **separate killable subprocess** with a hard `--task-timeout`
(SIGKILL). This is required: a flaky-gateway SSL read blocks at the C level and a
Python-thread watchdog CANNOT interrupt it — only killing the process frees it.

## Output (keyed by task_id, then mode)
```
<out>/trajectories.jsonl                 # combined, one line per attempt
<out>/<task_id>/<solve|refine>/
    trajectory.json    # replayable messages incl. thinking + tool calls + grade
    grade.json         # {n_correct, n_pairs, reward, compression_pct, fully_correct}
    solution.hy        # the final program the agent left
    raw_completions/    # faithful untransformed per-call LiteLLM I/O (the real CoT)
```

## Next (not yet built)
A **harvester**: filter to `fully_correct` (and `reward>0` for compression wins),
pick the best trajectory per task, convert Anthropic thinking → the student's expected
format, emit an SFT jsonl. Then SFT Qwen3.6-27B on the cluster (128k ctx / CP=8 / 8
nodes for the long trajectories).
