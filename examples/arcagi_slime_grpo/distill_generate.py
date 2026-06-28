#!/usr/bin/env python3
"""Local teacher-distillation generator: OpenHands + Claude (Sonnet, max thinking)
solve the synthetic ARC compression tasks, capturing full trajectories + CoT for SFT.

Why local (not on the cluster): the SFDC LLM Gateway resolves to a Salesforce-
internal VPC ingress that the Convergence HPC cluster cannot route to (curl times
out). Fraser's laptop / SFDC network CAN reach it. So teacher generation runs here;
only the downstream SFT training runs on the cluster. See memory
sfdc_gateway_claude_access.md.

Per task:
  1. Copy the task's auto-compress scaffold into an isolated working dir.
  2. Run the OpenHands SDK agent (terminal + file-editor tools), pointed at Claude
     via the SFDC gateway with extended thinking, on the same instruction the GRPO
     rollouts use — but with paths rewritten from the container's /opt/auto-compress
     to the local working dir + the local grader venv.
  3. Grade with score_synthetic.py --verify, parse the reward line.
  4. Persist the full event trajectory (messages incl. thinking + tool calls) and
     the raw per-call LiteLLM completion logs, tagged with the reward.

Harvest successful (reward>0, fully correct) trajectories into SFT data separately.

Usage (run with the grader+SDK deps available — see DISTILL_README or run via uv):
  SFDC_GATEWAY_KEY=sk-... python distill_generate.py --tasks 20 --out runs/distill1
  SFDC_GATEWAY_KEY=sk-... python distill_generate.py --task-ids 2x2-mosaic-recolored,foo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Fixed environment (validated 2026-06-28) ────────────────────────────────
SFDC_BASE_URL = (
    "https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl"
)
SFDC_CERT = os.path.expanduser("~/.aisuite/conf/npm-sfdc-certs.pem")
# Opus by default — Fraser wants the strongest teacher to maximize solve-rate on
# the harder ARC tasks (the 50 req/min limit applies equally to either model).
# Swap to anthropic/claude-sonnet-4-6 for cheaper/faster generation.
DEFAULT_MODEL = "anthropic/claude-opus-4-8"
# "Max thinking": a large extended-thinking budget. Sonnet caps well above this;
# 16k gives deep reasoning without runaway cost. Override with --thinking-budget.
DEFAULT_THINKING_BUDGET = 16384

CONTAINER_ROOT = "/opt/auto-compress"  # path baked into the GRPO task instruction

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTO_COMPRESS = Path(
    os.environ.get("AUTO_COMPRESS", str(Path.home() / "Projects" / "auto-compress"))
)
DEFAULT_PROMPT_DATA = Path(__file__).resolve().parent / "rl_tasks_train.jsonl"
# Local grader venv (hy+hyrule+numpy+scipy+arckit). Build once; see DISTILL_README.
DEFAULT_GRADER_PY = os.environ.get("ARCAGI_GRADER_PY", "/tmp/arcagi-grader/bin/python")

_RESULT_RE = re.compile(
    r"result:\s*(\d+)/(\d+)\s*correct.*?reward:\s*(-?[0-9.]+)", re.DOTALL
)
_COMPRESSION_RE = re.compile(r"compression:\s*([+-]?[0-9.]+)%")


def _setup_ssl() -> None:
    """Make every HTTP client (anthropic SDK + LiteLLM/httpx) trust the SFDC CA."""
    if not Path(SFDC_CERT).exists():
        sys.exit(f"SFDC cert bundle not found at {SFDC_CERT}")
    os.environ["SSL_CERT_FILE"] = SFDC_CERT
    os.environ["REQUESTS_CA_BUNDLE"] = SFDC_CERT


def _load_tasks(prompt_data: Path) -> list[dict]:
    rows = []
    with open(prompt_data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task_id(row: dict) -> str:
    md = row["metadata"]
    if isinstance(md, str):
        md = json.loads(md)
    return md["task_id"]


def _instruction(row: dict) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    # prompt is a list of chat messages; the user content holds the task.
    return prompt[0]["content"] if isinstance(prompt, list) else str(prompt)


def _localize_instruction(instruction: str, workdir: Path, grader_py: str) -> str:
    """Rewrite the container's /opt/auto-compress paths to the local working copy,
    and the bare `python` grader call to the local grader venv interpreter."""
    local_root = str(workdir)
    out = instruction.replace(CONTAINER_ROOT, local_root)
    # The instruction's verify block starts with a bare `python <root>/...`; pin it
    # to the grader venv so the agent's terminal uses an interpreter that has hy.
    out = out.replace(
        f"python {local_root}/sft/rl_tasks/score_synthetic.py",
        f"{grader_py} {local_root}/sft/rl_tasks/score_synthetic.py",
    )
    return out


# Distillation preamble: prepended to the base task instruction. Tells the teacher
# its turn budget (so trajectories stay within the student's training context) and
# to STUDY the grids / articulate the transformation before coding. We WANT the
# full agentic loop (study → implement → verify → read the delta → fix) distilled,
# including self-correction — not a one-shot answer. Getting fully correct is the
# priority; beating baseline (compression) is a bonus to pursue with remaining turns.
_DISTILL_PREAMBLE = """\
# How to work this task (read first)

You have a budget of {max_turns} turns. Work efficiently but DO think.

1. STUDY FIRST. Before writing any code, look carefully at every input/output
   pair and describe — in your own words — the exact structural transformation
   the task applies (sizes, tiling, recoloring, symmetry, per-region rules).
   State the pattern explicitly before you implement it.
2. IMPLEMENT a first version using the library primitives.
3. VERIFY after every edit with the score command below, and READ the per-pair
   grid DELTA when a pair is wrong — fix exactly what the delta shows.
4. Iterate study→edit→verify until it reports fully correct ({n_pairs}/{n_pairs}).
5. THEN, with any remaining turns, keep shrinking the program (fewer objects)
   while it stays fully correct — but never sacrifice correctness for size.

Getting it fully correct matters most. It is fine if you cannot beat the
baseline size — a correct solution is valuable. Stay within your turn budget.

---

"""

# Refine preamble (Type B): seed a NEW trajectory with a known-correct solution and
# ask the teacher to compress it. This is where compression wins reliably come from
# — solving-from-scratch rarely beats baseline, but refining a working solution does.
_REFINE_PREAMBLE = """\
# Refine an existing correct solution (read first)

You have a budget of {max_turns} turns.

A previous, fully-correct solution for this task already exists (shown below). It
scored {prior_size} objects (baseline {baseline_size}). Your job: make it SMALLER
(fewer objects under the count_objects metric) while keeping ALL pairs correct.

1. STUDY the existing solution: understand WHY it is correct and where its size
   comes from (redundant draws, literals that could be structural rewrites, etc.).
2. The file `sft/rl_tasks/{task_id}/{task_id}.hy` already contains this solution.
   Edit it to shrink it.
3. VERIFY after every edit with the score command below; never let it drop below
   fully correct ({n_pairs}/{n_pairs}). Read the grid DELTA if a pair breaks.
4. Iterate until it is correct AND smaller than {prior_size}.

## Existing correct solution ({prior_size} objects)

```hy
{prior_solution}
```

---

"""


def _compose_instruction(
    row: dict, workdir: Path, grader_py: str, max_turns: int,
    n_pairs: int, seed: dict | None = None,
) -> str:
    """Build the full agent instruction: distill (or refine) preamble + the
    localized base task instruction."""
    base = _localize_instruction(_instruction(row), workdir, grader_py)
    if seed is not None:
        pre = _REFINE_PREAMBLE.format(
            max_turns=max_turns,
            n_pairs=n_pairs,
            task_id=seed["task_id"],
            prior_size=seed["prior_size"],
            baseline_size=seed["baseline_size"],
            prior_solution=seed["prior_solution"],
        )
    else:
        pre = _DISTILL_PREAMBLE.format(max_turns=max_turns, n_pairs=n_pairs)
    return pre + base


def _task_hy_path(workdir: Path, task_id: str) -> Path:
    return workdir / "sft" / "rl_tasks" / task_id / f"{task_id}.hy"


def _baseline_size(row: dict) -> int | None:
    md = row["metadata"]
    if isinstance(md, str):
        md = json.loads(md)
    return md.get("baseline_size")


def _n_pairs(row: dict) -> int:
    """Number of input/output pairs (from the instruction's 'verify N pairs')."""
    instr = _instruction(row)
    m = re.search(r"reports?\s*`?(\d+)/(\d+)\s*correct", instr)
    if m:
        return int(m.group(2))
    # Fallback: count "### Pair" headers in the instruction.
    return max(1, len(re.findall(r"###\s*Pair", instr)))


def _prepare_workdir(auto_compress: Path, task_id: str, base_tmp: Path) -> Path:
    """Isolated working copy mirroring the real auto-compress layout.

    score_synthetic.py resolves REPO_ROOT = <its dir>.parents[1] and does
    `sys.path.insert(0, REPO_ROOT)`, then imports `eval` and (via the task .hy)
    `library`. So both eval.py and library.hy MUST live at the workdir ROOT
    (two levels above sft/rl_tasks/), not inside sft/rl_tasks/. We copy only the
    single task dir under sft/rl_tasks/ to keep it light (not all 1015)."""
    workdir = base_tmp / f"task_{task_id}"
    if workdir.exists():
        shutil.rmtree(workdir)
    src_rl = auto_compress / "sft" / "rl_tasks"
    dst_rl = workdir / "sft" / "rl_tasks"
    dst_rl.mkdir(parents=True)

    # Repo-root contract files (eval.py + library.hy) → workdir root.
    for fname in ("eval.py", "library.hy"):
        src = auto_compress / fname
        if not src.exists():  # some layouts keep them in sft/rl_tasks/
            src = src_rl / fname
        if src.exists():
            shutil.copy2(src, workdir / fname)

    # Grader script → sft/rl_tasks/ (where the instruction references it).
    shutil.copy2(src_rl / "score_synthetic.py", dst_rl / "score_synthetic.py")
    # library.hy is also imported by the task .hy via `(import library *)`; the
    # grader inserts the task dir on sys.path, so keep a copy beside the task too.
    lib = auto_compress / "library.hy"
    if not lib.exists():
        lib = src_rl / "library.hy"
    if lib.exists():
        shutil.copy2(lib, dst_rl / "library.hy")

    shutil.copytree(src_rl / task_id, dst_rl / task_id)
    return workdir


def _grade(grader_py: str, workdir: Path, task_id: str) -> dict:
    """Run the canonical --verify grader; parse correctness + reward + compression."""
    cmd = [
        grader_py,
        str(workdir / "sft" / "rl_tasks" / "score_synthetic.py"),
        "--root", str(workdir / "sft" / "rl_tasks"),
        "--verify", task_id,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = proc.stdout + "\n" + proc.stderr
    m = _RESULT_RE.search(out)
    n_correct = n = reward = None
    if m:
        n_correct, n, reward = int(m.group(1)), int(m.group(2)), float(m.group(3))
    cm = _COMPRESSION_RE.search(out)
    compression = float(cm.group(1)) if cm else None
    return {
        "n_correct": n_correct,
        "n_pairs": n,
        "reward": reward,
        "compression_pct": compression,
        "fully_correct": (n_correct is not None and n is not None and n_correct == n),
        "raw": out,
    }


def _run_one(row: dict, args, base_tmp: Path, out_dir: Path, seed: dict | None = None) -> dict:
    # Imports deferred so --help works without the SDK installed.
    from openhands.sdk import Agent, AgentContext, Conversation, LLM, Tool
    from openhands.sdk import LLMConvertibleEvent
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    task_id = _task_id(row)
    mode = "refine" if seed is not None else "solve"
    workdir = _prepare_workdir(args.auto_compress, task_id, base_tmp)
    n_pairs = _n_pairs(row)

    # Refine mode (Type B): overwrite the scaffold with the prior correct solution
    # so the agent starts from a working program and only needs to compress it.
    if seed is not None:
        _task_hy_path(workdir, task_id).write_text(seed["prior_solution"])

    instruction = _compose_instruction(
        row, workdir, args.grader_py, args.max_iterations, n_pairs, seed=seed
    )

    completions_dir = base_tmp / f"task_{task_id}" / "_completions"
    completions_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        api_key=os.environ["SFDC_GATEWAY_KEY"],
        base_url=SFDC_BASE_URL,
        extended_thinking_budget=args.thinking_budget,
        enable_encrypted_reasoning=True,
        max_output_tokens=args.max_output_tokens,
        # Per-call timeout + retry policy, tuned for the gateway's 50 req/min limit.
        # An OpenHands task makes ~20+ calls, so 429s are expected. litellm's retry
        # honors Retry-After; the limit resets at the next UTC minute, so we retry
        # PATIENTLY (up to ~70s/wait) to ride out a 429 rather than fail the task.
        # The orchestrator's subprocess SIGKILL deadline (--task-timeout) is the
        # real backstop, so patient retries here can't wedge the batch.
        timeout=90,            # bounds a genuinely stalled single read
        num_retries=4,
        retry_min_wait=5,
        retry_max_wait=70,     # > the 60s rate-limit reset window
        # Persist raw LiteLLM request/response per call — the faithful CoT capture.
        log_completions=True,
        log_completions_folder=str(completions_dir),
        usage_id=f"distill-{task_id}",
    )

    agent = Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        agent_context=AgentContext(),
    )

    events: list = []
    conversation = Conversation(
        agent=agent,
        workspace=str(workdir),
        max_iteration_per_run=args.max_iterations,
        callbacks=[lambda e: events.append(e)],
    )

    # NOTE on isolation: a flaky-gateway SSL read can block at the C level, which
    # a Python thread watchdog CANNOT interrupt (verified — conversation.close()
    # then wedges on the same dead socket). So the hard wall-clock guard lives in
    # the ORCHESTRATOR, which runs each task as a SEPARATE PROCESS and SIGKILLs it
    # past the deadline. Here (the worker) we just run to completion or natural
    # error; the parent process bounds our lifetime.
    t0 = time.time()
    error = None
    timed_out = False
    try:
        conversation.send_message(instruction)
        conversation.run()
    except Exception as e:  # capture, never abort
        error = f"{type(e).__name__}: {e}"
    finally:
        try:
            conversation.close()
        except Exception:
            pass
    elapsed = time.time() - t0

    grade = _grade(args.grader_py, workdir, task_id)

    # Capture the FINAL solution the agent left on disk — this is the artifact a
    # later refine pass (Type B) seeds from, and what the harvester reads.
    final_solution = None
    hy_path = _task_hy_path(workdir, task_id)
    if hy_path.exists():
        final_solution = hy_path.read_text()

    # Convert the OpenHands event stream into a replayable message list.
    try:
        llm_events = [e for e in events if isinstance(e, LLMConvertibleEvent)]
        messages = [m.model_dump() for m in LLMConvertibleEvent.events_to_messages(llm_events)]
    except Exception as e:
        messages = []
        error = (error + " | " if error else "") + f"events_to_messages: {e}"

    rec = {
        "task_id": task_id,
        "mode": mode,                       # "solve" | "refine"
        "model": args.model,
        "thinking_budget": args.thinking_budget,
        "max_turns": args.max_iterations,
        "elapsed_s": round(elapsed, 1),
        "timed_out": timed_out,
        "error": error,
        "grade": {k: v for k, v in grade.items() if k != "raw"},
        "final_solution": final_solution,
        "seed_size": (seed or {}).get("prior_size"),  # refine: size we started from
        "verify_output": grade["raw"][-4000:],
        "messages": messages,
        "n_events": len(events),
    }

    # Per-task artifacts dir, keyed unmistakably by task_id, then by mode (so a
    # refine attempt never clobbers the solve trajectory — each attempt is its own
    # independent trajectory). Survives the temp-dir cleanup: trajectory, grade,
    # the final .hy solution, and the RAW per-call LiteLLM CoT logs.
    task_out = out_dir / task_id / mode
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / "trajectory.json").write_text(json.dumps(rec, indent=2))
    (task_out / "grade.json").write_text(json.dumps(rec["grade"], indent=2))
    if final_solution is not None:
        (task_out / "solution.hy").write_text(final_solution)
    raw_dst = task_out / "raw_completions"
    if completions_dir.exists() and not raw_dst.exists():
        shutil.copytree(completions_dir, raw_dst)
    rec["task_out_dir"] = str(task_out)
    return rec


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output dir for trajectories")
    ap.add_argument("--tasks", type=int, default=20, help="how many tasks (first N)")
    ap.add_argument("--task-ids", type=str, default="", help="comma-separated explicit ids")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-output-tokens", type=int, default=32000)
    # Long attempts: user chose to allow ~50-turn trajectories (distill the full
    # study→implement→verify→self-correct loop). SFT on these wants 128k ctx (CP=8).
    ap.add_argument("--max-iterations", type=int, default=50)
    # Concurrency. The gateway limit is 50 req/min (per key, shared across workers).
    # Each worker is a SERIAL agent loop, so aggregate req/min ≈ parallel × 60/call_s.
    # Opus+thinking calls are ≥~6s, so 5 workers stay well under 50/min (~10-20/min
    # typical); patient 429-retry absorbs the rare burst. Raise cautiously.
    ap.add_argument("--parallel", type=int, default=5)
    # Re-attempt tasks even if a trajectory already exists (default: skip = one
    # attempt per task, resumable across re-runs).
    ap.add_argument("--force", action="store_true")
    # Hard per-task wall-clock cap (seconds), enforced by the orchestrator via
    # SIGKILL of the worker process. Bounds a stalled task so it can't wedge the
    # batch. Generous because the 50 req/min gateway limit means ~20-call tasks
    # legitimately spend time in patient 429 retries (~45 turns × call+possible
    # 60s rate-limit wait). The orchestrator runs tasks serially.
    ap.add_argument("--task-timeout", type=int, default=2400)
    ap.add_argument("--auto-compress", type=Path, default=DEFAULT_AUTO_COMPRESS)
    ap.add_argument("--prompt-data", type=Path, default=DEFAULT_PROMPT_DATA)
    ap.add_argument("--grader-py", default=DEFAULT_GRADER_PY)
    ap.add_argument(
        "--refine", action="store_true",
        help="after a task is solved fully-correct, run a SECOND independent "
             "trajectory seeded with that solution, asking the teacher to compress "
             "it further (Type B — where compression wins reliably come from).",
    )
    # Worker-mode flags (internal): the orchestrator re-invokes this script once
    # per task as a separate, killable process.
    ap.add_argument("--single", default="", help=argparse.SUPPRESS)  # task_id to run
    ap.add_argument("--single-mode", default="solve", help=argparse.SUPPRESS)  # solve|refine
    ap.add_argument("--seed-file", default="", help=argparse.SUPPRESS)  # refine seed json
    return ap


def _run_worker(args) -> int:
    """Worker mode: run exactly ONE task (one mode) and write its record. The
    parent process bounds our wall-clock via SIGKILL, so no in-process watchdog."""
    _setup_ssl()
    rows = _load_tasks(args.prompt_data)
    row = next((r for r in rows if _task_id(r) == args.single), None)
    if row is None:
        sys.exit(f"worker: task_id {args.single} not in {args.prompt_data}")
    seed = None
    if args.seed_file:
        seed = json.loads(Path(args.seed_file).read_text())
    with tempfile.TemporaryDirectory(prefix="arcagi_distill_") as tmp_s:
        _run_one(row, args, Path(tmp_s), args.out, seed=seed)
    return 0


def _launch_task(args, task_id: str, mode: str, seed: dict | None) -> dict:
    """Launch ONE task as a separate process (non-blocking). Returns a job handle
    dict the pool reaps later. The worker writes <out>/<task_id>/<mode>/
    trajectory.json; the orchestrator reads it back when the process exits."""
    task_out = args.out / task_id / mode
    rec_path = task_out / "trajectory.json"
    if rec_path.exists():
        rec_path.unlink()  # stale from a prior run

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single", task_id, "--single-mode", mode,
        "--out", str(args.out), "--model", args.model,
        "--thinking-budget", str(args.thinking_budget),
        "--max-output-tokens", str(args.max_output_tokens),
        "--max-iterations", str(args.max_iterations),
        "--auto-compress", str(args.auto_compress),
        "--prompt-data", str(args.prompt_data),
        "--grader-py", args.grader_py,
    ]
    if seed is not None:
        seed_file = task_out.parent / f"_seed_{mode}.json"
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps(seed))
        cmd += ["--seed-file", str(seed_file)]

    proc = subprocess.Popen(cmd, env=os.environ.copy())
    return {"task_id": task_id, "mode": mode, "proc": proc, "rec_path": rec_path,
            "t0": time.time(), "deadline": time.time() + args.task_timeout}


def _reap_record(job: dict, killed: bool, args) -> dict:
    """Build the trajectory record for a finished/killed job from its on-disk file
    (or a stub if the worker produced nothing)."""
    rec_path = job["rec_path"]
    elapsed = round(time.time() - job["t0"], 1)
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        if killed:
            rec["timed_out"] = True
            rec["error"] = (rec.get("error") or "") + f" | killed by orchestrator after {args.task_timeout}s"
        return rec
    return {
        "task_id": job["task_id"], "mode": job["mode"], "model": args.model,
        "elapsed_s": elapsed, "timed_out": killed,
        "error": (f"killed after {args.task_timeout}s" if killed
                  else f"worker exited rc={job['proc'].returncode} with no record"),
        "grade": {"n_correct": None, "n_pairs": None, "reward": None,
                  "compression_pct": None, "fully_correct": False},
        "final_solution": None, "messages": [], "n_events": 0,
    }


def _run_pool(args, jobspecs: list, emit) -> None:
    """Run jobspecs [(task_id, mode, seed), ...] through a pool of at most
    args.parallel concurrent subprocesses, refilling as each finishes. Each job
    has its own SIGKILL deadline so one stall never blocks the pool. Calls
    emit(rec) for every completed job (order = completion order)."""
    pending = list(jobspecs)
    running: list[dict] = []
    while pending or running:
        while pending and len(running) < args.parallel:
            tid, mode, seed = pending.pop(0)
            running.append(_launch_task(args, tid, mode, seed))
        # Poll running jobs; reap finished ones and SIGKILL any past deadline.
        still: list[dict] = []
        for job in running:
            rc = job["proc"].poll()
            if rc is not None:
                emit(_reap_record(job, killed=False, args=args))
            elif time.time() > job["deadline"]:
                job["proc"].kill()
                try:
                    job["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                emit(_reap_record(job, killed=True, args=args))
            else:
                still.append(job)
        running = still
        if running and len(running) >= args.parallel or (running and not pending):
            time.sleep(2)  # avoid a busy-spin while workers run


def _existing_solve(args, task_id: str) -> dict | None:
    """Resumability: if this task already has a solve trajectory on disk, return
    its grade record so we can skip it (one attempt per task) and still chain a
    refine pass from it."""
    p = args.out / task_id / "solve" / "trajectory.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _run_orchestrator(args) -> int:
    rows = _load_tasks(args.prompt_data)
    if args.task_ids:
        wanted = set(args.task_ids.split(","))
        rows = [r for r in rows if _task_id(r) in wanted]
    else:
        rows = rows[: args.tasks]

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "trajectories.jsonl"
    summary = []

    # APPEND to the combined jsonl so re-runs accumulate (resumable). Per-task
    # artifacts under <out>/<id>/<mode>/ are the source of truth for the harvester.
    out_f = open(results_path, "a")

    def _emit(rec: dict) -> dict:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        g = rec.get("grade", {})
        summary.append((rec["task_id"], rec["mode"], g.get("fully_correct"),
                        g.get("reward"), g.get("compression_pct")))
        print(
            f"  done {rec['task_id']} [{rec['mode']}] "
            f"correct={g.get('n_correct')}/{g.get('n_pairs')} "
            f"reward={g.get('reward')} compression={g.get('compression_pct')}% "
            f"({rec.get('elapsed_s')}s)"
            f"{' TIMEOUT' if rec.get('timed_out') else ''}"
            f"{' ERR' if rec.get('error') else ''}",
            flush=True,
        )
        return rec

    # ── Phase 1: SOLVE every not-yet-solved task (resumable, one attempt each) ──
    by_id = {_task_id(r): r for r in rows}
    todo, skipped = [], 0
    for tid in by_id:
        prev = _existing_solve(args, tid)
        if prev is not None and not args.force:
            skipped += 1  # already attempted — one attempt per task
        else:
            todo.append((tid, "solve", None))

    print(f"Distill {len(rows)} task(s) | model={args.model} "
          f"max_turns={args.max_iterations} parallel={args.parallel} "
          f"task_timeout={args.task_timeout}s")
    print(f"Phase 1 (solve): {len(todo)} to attempt, {skipped} already attempted (skipped)")
    if todo:
        _run_pool(args, todo, _emit)

    # ── Phase 2 (optional): REFINE each solved task to chase compression ──
    if args.refine:
        refine_jobs = []
        for tid, row in by_id.items():
            rec = _existing_solve(args, tid)
            if not (rec and rec.get("grade", {}).get("fully_correct") and rec.get("final_solution")):
                continue
            if (args.out / tid / "refine" / "trajectory.json").exists() and not args.force:
                continue
            g = rec["grade"]; base = _baseline_size(row)
            prior_size = base
            if base and g.get("compression_pct") is not None:
                prior_size = round(base * (1 - g["compression_pct"] / 100.0))
            refine_jobs.append((tid, "refine", {
                "task_id": tid, "prior_solution": rec["final_solution"],
                "prior_size": prior_size, "baseline_size": base,
            }))
        print(f"Phase 2 (refine): {len(refine_jobs)} solved task(s) to refine")
        if refine_jobs:
            _run_pool(args, refine_jobs, _emit)

    out_f.close()

    n_solve = sum(1 for _, m, c, _, _ in summary if m == "solve")
    n_solved = sum(1 for _, m, c, _, _ in summary if m == "solve" and c)
    n_win = sum(1 for _, _, c, r, _ in summary if c and r and r and r > 0)
    print(f"\n=== this run — solve attempts: {n_solved}/{n_solve} fully correct; "
          f"{n_win} trajectories with reward>0 (compression win) ===")
    print(f"combined trajectories -> {results_path}")
    print(f"per-task artifacts    -> {args.out}/<task_id>/<mode>/ (trajectory.json, grade.json, solution.hy, raw_completions/)")
    return 0


def main() -> int:
    args = _build_argparser().parse_args()

    if "SFDC_GATEWAY_KEY" not in os.environ:
        sys.exit("set SFDC_GATEWAY_KEY")
    if not Path(args.grader_py).exists():
        sys.exit(f"grader python not found: {args.grader_py} (build the grader venv first)")

    if args.single:
        return _run_worker(args)
    return _run_orchestrator(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
