# arcagi GRPO — debugging state (2026-06-25)

Facts only — what was directly observed, and what is still unknown. Written
after a long debugging session that went in circles; this is meant to be the
trustworthy ground truth to restart from.

## The symptom that blocks training

Every GRPO training run reaches the rollout phase, runs thousands of agent
sessions, but **essentially every session ends `status=ERROR`,
`error="no completions"`, `record_count=0`, `traces=[]`**. With no usable
traces, GRPO groups are dropped ("zero trainable tokens") and the optimizer has
nothing to learn from. Training does not progress.

Observed across jobs 19516 (2-node), 19524 (4-node), 19529 (1-node smoke).

## RESOLVED (2026-06-25, jobs 19536 + 19537) — "no completions" was a tmux env crash

The blocker is fixed and proven. Root cause: OpenHands' `TerminalTool` copies
the process env into its tmux session via `tmux set-environment` (per-value
length limit); the preset passed the whole task prompt as `AGENT_INSTRUCTION`
(~54 KB/row), overflowing tmux → `ValueError: command too long` → the agent
died in tool init ~4 s in, BEFORE its first LLM call. That is the entire reason
for "0 `←` lines / SGLang only health-pings / no completions". Serving and
loopback were healthy all along.

- **Fix:** `openhands_sdk.py` runner now does `os.environ.pop("AGENT_INSTRUCTION")`
  (was `.get`). Instruction still reaches the model via `send_message`; it's no
  longer left in the env for tmux to copy.
- **Diagnostic enablers that made it visible (keep these):** `set -o pipefail`
  in the preset (crash fails the step instead of being masked by `tee`), and
  `POLAR_KEEP_SESSION_DIRS=1` (preserves `logs/agent/openhands-sdk.txt` past
  teardown — but note preserved dirs land under node-local
  `/mnt/localssd/tmp/<jobid>/`, gone when the node releases; read them from the
  NFS job log instead).
- **Proof (19537):** 31 proxied `← POST /v1/chat/completions`, session
  `record_count=22`, `trace_count=22`, all chains reconstructed full.

### Context-overflow failure — ALSO RESOLVED (job 19538)

19537's only remaining failure was the agent overrunning the 40000 context on a
long conversation (`input 40096 > 40000` → SGLang 400 → ERROR). Fixed by tuning,
informed by measurements off 19537's captured completions:
- per-turn prompt growth ≈ **600 tok** off a ~**25K instruction floor**; one
  turn spiked ~9.6K (a big tool/eval output) — that spike, not turn count, blew
  the cap. 22 turns only reached ~30K.
- the SGLang KV pool is sized by `--sglang-mem-fraction-static` (≈**1.69M
  tokens** at 0.8/TP=4), NOT by context length, so raising context is ~free;
  model ceiling is `max_position_embeddings=262144`.

**Settings applied** (run.sh + both polar_configs + smoke job):
`SGLANG_CONTEXT_LENGTH 40000→65536`, `--rollout-max-prompt-len 32000→49152`
(it's a lower cap than context, so it had to move too), `max_iterations 30→15`.

**Job 19538 result (1-node smoke, openhands_sdk):** both sessions
`status=COMPLETED`, `error=None`, 0 context-overflow 400s, traces captured,
run 14–28 s. (Reward 1.0 here is the `session_completed` smoke evaluator —
completion, NOT compression quality. The graded ArcCompressEvaluator reward is
exercised only in the real training configs.)

### State now: plumbing fully unblocked

The original "no completions" blocker AND the follow-on context-overflow are both
resolved on 1 node. Next step is a real training run (`run.sh` via the slurm
wrapper). Watch for: (1) whether GRPO groups now have trainable tokens (they
should), (2) the real compression reward distribution, (3) whether the 27B
train-side memory split holds at the new context length (serve-side KV pool is
unaffected; only confirm the train ranks).

## BREAKTHROUGH (2026-06-25, code trace — supersedes the topology theory below)

A read of the actual code path overturns the two leading hypotheses. **This is
not a multi-node / topology / loopback bug.**

- **Containers ALWAYS run on the gateway's node.** `DockerRuntime.start()` runs
  `docker create`/`start` as a *local subprocess* (`runtime/base.py:147`,
  `asyncio.create_subprocess_exec`) inside the `GatewayNodeManager` process.
  There is exactly ONE gateway node in the topology (`localhost-node-01`) and
  `run.sh:175` starts `serve_gateway` only on node0. So every agent container is
  spawned on node0, co-located with the gateway at `127.0.0.1:8100`, with
  `--network host`. Loopback cannot fail due to node placement. Hypothesis #2
  (containers on a non-gateway node) is structurally impossible.
- **The failure reproduces on 1 node (19529).** A placement/topology bug can't
  reproduce single-node. That alone kills the topology theory. → No per-node
  gateways, no rendered tailscale URLs needed.
- **The real bug: the agent crashes ~4 s in, before its first LLM call, and the
  crash is SILENTLY SWALLOWED by two code defects:**
  1. `| tee` masks the runner's exit code. The harness runs
     `"$PYTHON_BIN" runner.py 2>&1 | tee …openhands-sdk.txt` under `bash -lc`
     with no `pipefail` (`agent/presets/openhands_sdk.py:81`,
     `runtime/docker.py:149`). Pipeline status = tee's = 0, so a crashing runner
     is recorded as a **`completed`** step → builder finds 0 completions →
     `error="no completions"`. That symptom is a DECOY for the real traceback.
  2. The evidence is deleted every run. The agent's stdout lives at
     `session_dir/logs/agent/openhands-sdk.txt` (bind-mounted), and
     `_handle_postrun` `shutil.rmtree`s the whole session dir on teardown
     (`node.py:555`). Every prior debug session read gateway/SGLang logs — never
     this file, because it was already gone.

**Fixes applied this session (need a smoke run to confirm):**
- `openhands_sdk.py`: prepend `set -o pipefail;` so a runner crash fails the
  step (status `failed`, real exit code) instead of silently `completed`.
- `node.py`: `_remove_session_dir_best_effort` now skips deletion when
  `POLAR_KEEP_SESSION_DIRS` is set (default unchanged = delete).
- `job_arcagi_rl_smoke.slurm`: exports `POLAR_KEEP_SESSION_DIRS=1` and dumps
  `/tmp/session-*/logs/agent/openhands-sdk.txt` + step stdout in RESULTS.

**Next action: run `sbatch job_arcagi_rl_smoke.slurm` and read the agent stdout.**
It will, for the first time, show WHY the OpenHands runner dies in 4 s (likely
candidates: a litellm/model-name validation error on
`openai/Fraser/Qwen3.6-27B-ARC-Hy`, an SDK import/version mismatch, or an early
config exception — none of which are LLM-connectivity issues).

## What is CONFIRMED WORKING (each tested directly)

- **SGLang serving the SFT'd 27B.** Patched sglang `0.5.13.post1`, TP=4,
  `--disable-custom-all-reduce`. Comes up healthy; `/v1/chat/completions`
  returns 200.
  - `max_tokens=16` → truncated, all `reasoning_content`, empty `content`.
  - `max_tokens=1500` → `finish_reason=stop`, closes the thinking block, emits
    `content: "36"` (correct). So the model generates fine; the 16-token empty
    was pure truncation, not thinking-runaway.
  - The token-metadata patch's logprobs path works: returns `token_id` per token.
- **The SGLang token-metadata patch is NOT broken.** Generation + logprobs both
  work post-patch. (Earlier I blamed it on timeline correlation — that was wrong.)
- **The OpenHands agent, pointed DIRECTLY at SGLang** (LLM_BASE_URL =
  `http://<sglang>:port/v1`), runs a real conversation and replies correctly
  (`Message from Agent: Ready.`, input 5.82K tok, output 20). So the agent,
  the rl docker image, the baked openhands-sdk 1.17 venv, and all tool imports
  (Terminal, FileEditor, TaskTracker) are fine.
- **The training stack** (slime v0.3.0 + Megatron commit
  `1dcf0dafa884ad52ffb243625717a3471643e087` + TE 2.10.0 + FLA), weight
  conversion (HF→torch_dist, 51 GB), and multi-node Ray bringup all work — the
  run gets all the way to rollouts and even did weight-sync steps.
- **The synthetic rl_tasks pipeline** (1015 tasks): scaffolding, `score_synthetic.py`
  (`--check`/`--size`), JSONL build, and the rl image (all task .hy reset to
  baseline) are validated in-container.

## The one FAILING hop: agent → gateway proxy

- In a real rollout, the agent's LLM request **never reaches the gateway proxy**.
  The gateway logs an incoming proxied request as a line starting with `← `;
  the count of those is **0** in every gateway log checked (19524, 19529).
- SGLang's own logs in the training runs show only `#new-token: 1` prefills
  (health pings) — **never a real ~13K-token agent prompt**. Corroborates that
  no real generation request ever flows.
- The agent bails in ~4 s (`run_ms ≈ 3900`), before iteration 1.

## What I have NOT cleanly proven (open — do not trust prior claims)

- ~~**Whether `127.0.0.1:8100` is reachable from the agent container.**~~
  LARGELY MOOT per the breakthrough above: container is `--network host` on the
  gateway node, so loopback is the host's loopback. Still worth a one-line
  in-container `curl 127.0.0.1:8100/health` in the smoke run for total
  certainty, but connectivity is no longer the suspected cause — the agent dies
  before it ever issues an LLM call.
- ~~**Whether multi-node rollout containers landed on the gateway node.**~~
  RESOLVED: they always do (the gateway spawns them as local subprocesses).
  See breakthrough above.

## Known architectural facts (from code/docs, not yet load-bearing-tested here)

- Gateway injects `OPENAI_BASE_URL = {node.public_url}/v1` to the agent
  (`gateway/node.py:769`) and binds `state.node.host` (`gateway/server.py:825`).
  In our topology both are `127.0.0.1`.
- Gateway README: "per-worker FastAPI service ... proxies the agent's LLM calls
  to a **local** inference server." Designed for agent+gateway+inference
  co-located on one node.
- Two topology modes exist:
  - **rollout-only** (`examples/swebench_verified/topology.sgl.yaml`): you run
    SGLang engines, list each as a gateway node `base_url` (all `127.0.0.1` there
    because single-box).
  - **slime-training** (our `topology.yaml`): slime owns the engines behind a
    router on :9000; gateway `base_url: ${SGLANG_ROUTER_BASE_URL}` is rendered to
    a routable `detect_host_ip():9000` at launch.
- Cluster nodes are reachable over tailscale by hostname, e.g.
  `http://slurm1-a3nodeset-10:8002/v1`. Node names are only known at job time
  (from `scontrol show hostnames $SLURM_JOB_NODELIST`), so any routable topology
  must be rendered at launch (as we already do for the router URL).

## The memory constraint that drove the multi-node detour

- 27B at **TP=4 on one node** OOMs during MegatronTrainRayActor init: each rank
  held ~10.9 B params; a ~40.6 GB fp32 grad/master allocation failed with ~36 GB
  free. Optimizer CPU offload alone did NOT fix it (the failing alloc precedes
  offload).
- **TP=8 across a full node fit** (no OOM) — which is why the runs went
  multi-node (node0 train TP=8, other nodes serve). That split is what put the
  rollout containers potentially on non-gateway nodes.
- Polar's intended shape is single-node **colocated** (train+serve+gateway+agents
  together; the swegym example uses slime `--colocate` + offload). We have not
  tried colocated-on-one-node for 27B.

## Correction to earlier claims in this session

- I earlier said the 2-node run "19516 worked / produced real completions." That
  was **wrong** — its SGLang log shows a single health-ping prefill and
  `no_usable=8208`. 19516 failed the same way as every other run. The
  "completed with N results" log lines were sessions completing with EMPTY
  trajectories, not real generations.

## Cleanest next diagnostic (not yet run)

A faithful loopback test: launch the gateway bound to `127.0.0.1:8100`, then run
the agent container (`--network host`, same node) with the EXACT
`OPENAI_BASE_URL=http://127.0.0.1:8100/v1` Polar injects, and check whether a
`← ` line appears in the gateway log.
- `← ` appears → co-located loopback works; the multi-node failure is purely
  that containers ran on a node without a gateway → fix = per-node gateway +
  rendered routable (tailscale) URLs, OR run colocated.
- still fails → the bug is not topology/placement and is deeper than currently
  understood.

## Jobs / artifacts

- No GPU jobs should be running (last runs cancelled).
- rl image tarball: `/home/fraser_convergence_ai/arcagi-image/polar-arcagi-rl.tar.gz`
- arcagi image tarball: `/home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz`
- torch_dist ckpt: `~/ProRL-Agent-Server/tmp/checkpoints/Qwen3.6-27B-ARC-Hy_torch_dist`
- rl JSONL: `examples/arcagi_slime_grpo/rl_tasks_train.jsonl` (1015 rows, on cluster)
- W&B project: `polar-arcagi-grpo` (runs are empty of useful signal so far).
