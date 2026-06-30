#!/usr/bin/env python3
"""Teacher-distillation generator for the REAL ARC-AGI tasks (not the synthetic
rl_tasks). OpenHands + Claude (Opus, max thinking) solve+compress each task from a
BASELINE scaffold, capturing full trajectories + CoT for SFT.

Differs from distill_generate.py (synthetic rl_tasks) only in the task-specific
pieces; the orchestration (subprocess-per-task isolation, SIGKILL watchdog,
parallel pool, resumable, refine pass, full-CoT capture, SFDC-gateway/Opus LLM) is
the same proven machinery, inlined here.

Task source: every <id>.hy under <auto-compress>/tasks/{arcagi,arcagi2}/{train,eval}/
(~1551 tasks). Each is scaffolded to BASELINE (literal `(quote (grid [[...]]))`
inputs + stub `to-output`) — the agent solves from the stub, exactly like the
synthetic pipeline.

Per task (in an isolated COPY of the auto-compress repo so eval.py + library.hy +
the task resolve):
  1. scaffold the task .hy to baseline (eval.py --scaffold form).
  2. agent: study grids → implement to-output → `eval.py --task <ver>/<id>` (shows
     per-pair grid DELTA on mismatch) → fix → shrink. Verify loop drives it.
  3. grade: parse `eval.py --task` (N/N correct) + `eval.py --size` (object count);
     reward vs the RAW-ARRAY BASELINE (literal input + literal output cells) =
     max(0, (raw_baseline - size)/raw_baseline) when fully correct, else -1.
  4. capture trajectory; --refine seeds a 2nd pass from the solved solution.

Why local: the SFDC gateway is unreachable from the HPC cluster (see memory
sfdc_gateway_claude_access.md). Generation runs on Fraser's machine.

Usage (run via uv with the SDK + arcagi grader venv available):
  SFDC_GATEWAY_KEY=sk-... AUTO_COMPRESS=~/Projects/auto-compress \
  ARCAGI_GRADER_PY=/tmp/arcagi-grader/bin/python \
  uv run --with openhands-sdk --with openhands-tools --with litellm --with certifi \
    python distill_generate_arcagi.py --out runs/distill_arcagi --scope all --refine
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

# ── Fixed environment (validated 2026-06-28..30) ────────────────────────────
SFDC_BASE_URL = (
    "https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl"
)
SFDC_CERT = os.path.expanduser("~/.aisuite/conf/npm-sfdc-certs.pem")
DEFAULT_MODEL = "anthropic/claude-opus-4-8"
DEFAULT_THINKING_BUDGET = 16384

DEFAULT_AUTO_COMPRESS = Path(
    os.environ.get("AUTO_COMPRESS", str(Path.home() / "Projects" / "auto-compress"))
)
DEFAULT_GRADER_PY = os.environ.get("ARCAGI_GRADER_PY", "/tmp/arcagi-grader/bin/python")

# (version, split) pairs that make up each scope.
SCOPES = {
    "all":          [("arcagi", "train"), ("arcagi", "eval"),
                     ("arcagi2", "train"), ("arcagi2", "eval")],
    "arcagi-train": [("arcagi", "train")],
    "arcagi-eval":  [("arcagi", "eval")],
    "arcagi2-train":[("arcagi2", "train")],
    "arcagi2-eval": [("arcagi2", "eval")],
    "arcagi":       [("arcagi", "train"), ("arcagi", "eval")],
    "arcagi2":      [("arcagi2", "train"), ("arcagi2", "eval")],
}

_RESULT_RE = re.compile(r"([0-9]+)\s*/\s*([0-9]+)\s*correct")


def _setup_ssl() -> None:
    if not Path(SFDC_CERT).exists():
        sys.exit(f"SFDC cert bundle not found at {SFDC_CERT}")
    os.environ["SSL_CERT_FILE"] = SFDC_CERT
    os.environ["REQUESTS_CA_BUNDLE"] = SFDC_CERT


# ── Task enumeration ────────────────────────────────────────────────────────
# task_id here is "<version>/<id>" (e.g. "arcagi/007bbfb7"), matching eval.py's
# --task / --size argument. The on-disk path is tasks/<ver>/<split>/<id>.hy.

def _list_tasks(auto_compress: Path, scope: str) -> list[dict]:
    pairs = SCOPES[scope]
    out = []
    for ver, split in pairs:
        d = auto_compress / "tasks" / ver / split
        if not d.is_dir():
            continue
        for hy in sorted(d.glob("*.hy")):
            out.append({"task_id": f"{ver}/{hy.stem}", "version": ver,
                        "split": split, "id": hy.stem})
    return out


def _safe_id(task_id: str) -> str:
    """Filesystem-safe per-task dir name ('arcagi/007bbfb7' -> 'arcagi__007bbfb7')."""
    return task_id.replace("/", "__")


def _task_hy_relpath(version: str, split: str, tid: str) -> str:
    return f"tasks/{version}/{split}/{tid}.hy"


# ── Workdir: isolated copy of the auto-compress repo ────────────────────────

def _prepare_workdir(auto_compress: Path, row: dict, base_tmp: Path) -> Path:
    """Isolated working copy holding everything eval.py needs to score ONE task:
    eval.py + library.hy at the root, and tasks/<ver>/<split>/<id>.hy. We copy the
    whole tasks/<ver>/<split> dir's single task file (not all tasks) plus the repo
    contract files, so `eval.py --task <ver>/<id>` resolves exactly as in-repo."""
    ver, split, tid = row["version"], row["split"], row["id"]
    workdir = base_tmp / f"task_{_safe_id(row['task_id'])}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    for fname in ("eval.py", "library.hy"):
        src = auto_compress / fname
        if src.exists():
            shutil.copy2(src, workdir / fname)
    # arckit metadata: eval.py loads grids via arckit (installed in the grader venv),
    # so we don't need the dataset files — but the task .hy must exist at the path.
    dst = workdir / "tasks" / ver / split
    dst.mkdir(parents=True)
    shutil.copy2(auto_compress / "tasks" / ver / split / f"{tid}.hy", dst / f"{tid}.hy")
    return workdir


def _task_hy_path(workdir: Path, row: dict) -> Path:
    return workdir / _task_hy_relpath(row["version"], row["split"], row["id"])


# ── Scaffolding + grids + baseline size (via the grader venv, which has arckit) ──

def _grader_eval(grader_py: str, workdir: Path, args_list: list[str], timeout: int = 120):
    """Run `python eval.py <args>` inside the workdir with the grader venv."""
    proc = subprocess.run(
        [grader_py, str(workdir / "eval.py"), *args_list],
        capture_output=True, text=True, timeout=timeout, cwd=str(workdir),
    )
    return proc.stdout + "\n" + proc.stderr


def _scaffold_and_describe(grader_py: str, workdir: Path, row: dict) -> dict:
    """Reset the task .hy to BASELINE scaffold and pull the task's grids + the
    raw-array baseline object count. Done in one helper script run in the grader
    venv (which has arckit + hy + eval.py's helpers)."""
    ver, tid = row["version"], row["id"]
    helper = workdir / "_scaffold_describe.py"
    helper.write_text(_SCAFFOLD_HELPER)
    out = subprocess.run(
        [grader_py, str(helper), ver, tid],
        capture_output=True, text=True, timeout=120, cwd=str(workdir),
    )
    txt = out.stdout
    # The helper prints a JSON line prefixed with @@DESC@@ for robust parsing.
    desc = {}
    for line in txt.splitlines():
        if line.startswith("@@DESC@@"):
            desc = json.loads(line[len("@@DESC@@"):])
            break
    if not desc:
        desc = {"error": (out.stdout + out.stderr)[-2000:]}
    return desc


def _grade(grader_py: str, workdir: Path, row: dict, raw_baseline: int) -> dict:
    """Score the task: correctness via `eval.py --task`, size via `eval.py --size`,
    reward vs the raw-array baseline. Mirrors the synthetic reward semantics."""
    tid = row["task_id"]
    task_out = _grader_eval(grader_py, workdir, ["--task", tid])
    m = _RESULT_RE.search(task_out)
    n_correct = n = None
    if m:
        n_correct, n = int(m.group(1)), int(m.group(2))
    fully_correct = (n_correct is not None and n is not None and n_correct == n)

    size = None
    size_out = _grader_eval(grader_py, workdir, ["--size", tid])
    for line in size_out.splitlines():
        s = line.strip()
        if s.isdigit():
            size = int(s)
            break

    reward = None
    compression_pct = None
    if fully_correct and size is not None and raw_baseline and raw_baseline > 0:
        reward = max(0.0, min(1.0, (raw_baseline - size) / raw_baseline))
        compression_pct = (1.0 - size / raw_baseline) * 100.0
    elif not fully_correct:
        reward = -1.0

    return {
        "n_correct": n_correct, "n_pairs": n, "fully_correct": fully_correct,
        "size": size, "raw_baseline": raw_baseline,
        "reward": reward, "compression_pct": compression_pct,
        "raw": (task_out + "\n--- size ---\n" + size_out),
    }


# ── Prompt ──────────────────────────────────────────────────────────────────

_PREAMBLE = """\
You are compressing one ARC-AGI task into a structural Hy program that shares a
common `library.hy`. You have a budget of {max_turns} turns.

# Goal
Edit `{hy_relpath}` so that `(to-output (inputs)[i])` reproduces output i for every
pair, while the program is as SMALL as possible under the count_objects metric (a
literal grid costs one object per cell; a named constructor call costs ~1). The
file currently holds the BASELINE scaffold: literal `(quote (grid [[...]]))` inputs
and a stub `to-output` that must be replaced with a real structural rewrite.

# How to work (read first)
1. STUDY FIRST. Look at every input/output pair below and state, in your own words,
   the exact structural transformation (sizes, tiling, recoloring, symmetry, masks,
   per-region rules) BEFORE writing code.
2. IMPLEMENT `to-output` using the `library.hy` primitives (and rewrite `inputs()`
   constructively if that is smaller). If you use hyrule macros (->, cond, when,
   for, ...), start the file with `(require hyrule * :readers *)`.
3. VERIFY after every edit by running this EXACT command from the terminal:

   {grader_py} {workdir}/eval.py --task {task_id}

   It prints per-pair correctness and, on a wrong pair, a grid DELTA marking each
   off cell — fix exactly what the delta shows. To check your size, run:

   {grader_py} {workdir}/eval.py --size {task_id}

4. Iterate study→edit→verify until `eval.py --task` reports all pairs correct, then
   keep shrinking the object count while it STAYS fully correct. Do NOT import or
   hy.eval the file yourself — use the eval.py commands.

Getting fully correct matters most; smaller-than-baseline is a bonus to pursue with
remaining turns. The raw-array baseline for this task is {raw_baseline} objects.

# Task {task_id}  ({n_pairs} pairs)

## Grids
{grids}

## Current body of `{hy_relpath}` (baseline scaffold)
```hy
{scaffold}
```

## `library.hy` (shared primitives — read-only)
```hy
{library}
```
"""

_REFINE_PREAMBLE = """\
You are compressing one ARC-AGI task. You have a budget of {max_turns} turns.

A previous, fully-correct solution for `{hy_relpath}` already exists (below). It
scored {prior_size} objects (raw-array baseline {raw_baseline}). Make it SMALLER
under the count_objects metric while keeping ALL pairs correct.

1. STUDY the existing solution: understand why it is correct and where its size
   comes from (literal data that could be structural, redundant draws, ...).
2. The file already contains this solution — edit it to shrink it.
3. VERIFY after every edit:  {grader_py} {workdir}/eval.py --task {task_id}
   (never drop below fully correct), and  --size {task_id}  to track objects.
4. Iterate until it is correct AND smaller than {prior_size}.

# Task {task_id}  ({n_pairs} pairs)

## Grids
{grids}

## Existing correct solution ({prior_size} objects)
```hy
{prior_solution}
```

## `library.hy` (shared primitives — read-only)
```hy
{library}
```
"""


def _compose_instruction(row, workdir, grader_py, max_turns, desc, seed):
    library = (workdir / "library.hy").read_text()
    common = dict(
        max_turns=max_turns, hy_relpath=_task_hy_relpath(row["version"], row["split"], row["id"]),
        grader_py=grader_py, workdir=str(workdir), task_id=row["task_id"],
        n_pairs=desc.get("n_pairs", "?"), raw_baseline=desc.get("raw_baseline", "?"),
        grids=desc.get("grids", "(grids unavailable)"), library=library,
    )
    if seed is not None:
        return _REFINE_PREAMBLE.format(
            prior_size=seed["prior_size"], prior_solution=seed["prior_solution"], **common
        )
    return _PREAMBLE.format(scaffold=desc.get("scaffold", ""), **common)


# ── The single-task worker body ─────────────────────────────────────────────

def _run_one(row: dict, args, base_tmp: Path, out_dir: Path, seed: dict | None = None) -> dict:
    from openhands.sdk import Agent, AgentContext, Conversation, LLM, Tool
    from openhands.sdk import LLMConvertibleEvent
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    task_id = row["task_id"]
    mode = "refine" if seed is not None else "solve"
    workdir = _prepare_workdir(args.auto_compress, row, base_tmp)

    # Scaffold to baseline (solve) and pull grids + raw-array baseline size.
    desc = _scaffold_and_describe(args.grader_py, workdir, row)
    raw_baseline = desc.get("raw_baseline") or 0
    n_pairs = desc.get("n_pairs")

    # Refine mode: overwrite the baseline scaffold with the prior correct solution.
    if seed is not None:
        _task_hy_path(workdir, row).write_text(seed["prior_solution"])

    instruction = _compose_instruction(row, workdir, args.grader_py,
                                        args.max_iterations, desc, seed)

    completions_dir = base_tmp / f"task_{_safe_id(task_id)}" / "_completions"
    completions_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model, api_key=os.environ["SFDC_GATEWAY_KEY"], base_url=SFDC_BASE_URL,
        extended_thinking_budget=args.thinking_budget, enable_encrypted_reasoning=True,
        max_output_tokens=args.max_output_tokens,
        timeout=90, num_retries=4, retry_min_wait=5, retry_max_wait=70,
        log_completions=True, log_completions_folder=str(completions_dir),
        usage_id=f"distill-{_safe_id(task_id)}",
    )
    agent = Agent(llm=llm,
                  tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
                  agent_context=AgentContext())
    events: list = []
    conversation = Conversation(agent=agent, workspace=str(workdir),
                                max_iteration_per_run=args.max_iterations,
                                callbacks=[lambda e: events.append(e)])
    t0 = time.time()
    error = None
    try:
        conversation.send_message(instruction)
        conversation.run()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        try:
            conversation.close()
        except Exception:
            pass
    elapsed = time.time() - t0

    grade = _grade(args.grader_py, workdir, row, raw_baseline)

    final_solution = None
    hy_path = _task_hy_path(workdir, row)
    if hy_path.exists():
        final_solution = hy_path.read_text()

    try:
        llm_events = [e for e in events if isinstance(e, LLMConvertibleEvent)]
        messages = [m.model_dump() for m in LLMConvertibleEvent.events_to_messages(llm_events)]
    except Exception as e:
        messages = []
        error = (error + " | " if error else "") + f"events_to_messages: {e}"

    rec = {
        "task_id": task_id, "mode": mode, "version": row["version"], "split": row["split"],
        "model": args.model, "thinking_budget": args.thinking_budget,
        "max_turns": args.max_iterations, "elapsed_s": round(elapsed, 1),
        "timed_out": False, "error": error,
        "grade": {k: v for k, v in grade.items() if k != "raw"},
        "final_solution": final_solution,
        "seed_size": (seed or {}).get("prior_size"),
        "verify_output": grade["raw"][-4000:], "messages": messages, "n_events": len(events),
    }
    task_out = out_dir / _safe_id(task_id) / mode
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


# ── Orchestration (subprocess pool + SIGKILL watchdog; same as the synthetic one) ──

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scope", default="all", choices=sorted(SCOPES))
    ap.add_argument("--tasks", type=int, default=0, help="limit to first N (0 = all in scope)")
    ap.add_argument("--task-ids", default="", help="comma-separated <ver>/<id> subset")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    ap.add_argument("--max-output-tokens", type=int, default=32000)
    ap.add_argument("--max-iterations", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--task-timeout", type=int, default=2400)
    ap.add_argument("--auto-compress", type=Path, default=DEFAULT_AUTO_COMPRESS)
    ap.add_argument("--grader-py", default=DEFAULT_GRADER_PY)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--single", default="", help=argparse.SUPPRESS)
    ap.add_argument("--single-mode", default="solve", help=argparse.SUPPRESS)
    ap.add_argument("--seed-file", default="", help=argparse.SUPPRESS)
    return ap


def _row_for(args, task_id: str) -> dict | None:
    for r in _list_tasks(args.auto_compress, "all"):
        if r["task_id"] == task_id:
            return r
    return None


def _run_worker(args) -> int:
    _setup_ssl()
    row = _row_for(args, args.single)
    if row is None:
        sys.exit(f"worker: task {args.single} not found")
    seed = json.loads(Path(args.seed_file).read_text()) if args.seed_file else None
    with tempfile.TemporaryDirectory(prefix="arcagi_distill_") as tmp_s:
        _run_one(row, args, Path(tmp_s), args.out, seed=seed)
    return 0


def _launch_task(args, task_id: str, mode: str, seed: dict | None) -> dict:
    task_out = args.out / _safe_id(task_id) / mode
    rec_path = task_out / "trajectory.json"
    if rec_path.exists():
        rec_path.unlink()
    cmd = [sys.executable, os.path.abspath(__file__),
           "--single", task_id, "--single-mode", mode, "--out", str(args.out),
           "--model", args.model, "--thinking-budget", str(args.thinking_budget),
           "--max-output-tokens", str(args.max_output_tokens),
           "--max-iterations", str(args.max_iterations),
           "--auto-compress", str(args.auto_compress), "--grader-py", args.grader_py]
    if seed is not None:
        seed_file = task_out.parent / f"_seed_{mode}.json"
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps(seed))
        cmd += ["--seed-file", str(seed_file)]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    return {"task_id": task_id, "mode": mode, "proc": proc, "rec_path": rec_path,
            "t0": time.time(), "deadline": time.time() + args.task_timeout}


def _reap_record(job: dict, killed: bool, args) -> dict:
    rec_path = job["rec_path"]
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        if killed:
            rec["timed_out"] = True
            rec["error"] = (rec.get("error") or "") + f" | killed after {args.task_timeout}s"
        return rec
    return {"task_id": job["task_id"], "mode": job["mode"], "model": args.model,
            "elapsed_s": round(time.time() - job["t0"], 1), "timed_out": killed,
            "error": (f"killed after {args.task_timeout}s" if killed
                      else f"worker rc={job['proc'].returncode} no record"),
            "grade": {"n_correct": None, "n_pairs": None, "reward": None,
                      "compression_pct": None, "fully_correct": False},
            "final_solution": None, "messages": [], "n_events": 0}


def _run_pool(args, jobspecs: list, emit) -> None:
    pending = list(jobspecs)
    running: list[dict] = []
    while pending or running:
        while pending and len(running) < args.parallel:
            tid, mode, seed = pending.pop(0)
            running.append(_launch_task(args, tid, mode, seed))
        still = []
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
        if running:
            time.sleep(2)


def _existing(args, task_id: str, mode: str) -> dict | None:
    p = args.out / _safe_id(task_id) / mode / "trajectory.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _run_orchestrator(args) -> int:
    rows = _list_tasks(args.auto_compress, args.scope)
    if args.task_ids:
        want = set(args.task_ids.split(","))
        rows = [r for r in rows if r["task_id"] in want]
    if args.tasks:
        rows = rows[: args.tasks]
    if not rows:
        print("no tasks in scope"); return 1

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "trajectories.jsonl"
    out_f = open(results_path, "a")
    summary = []

    def _emit(rec):
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        g = rec.get("grade", {})
        summary.append((rec["task_id"], rec["mode"], g.get("fully_correct"), g.get("reward")))
        print(f"  done {rec['task_id']} [{rec['mode']}] "
              f"correct={g.get('n_correct')}/{g.get('n_pairs')} reward={g.get('reward')} "
              f"comp={g.get('compression_pct')}% ({rec.get('elapsed_s')}s)"
              f"{' TIMEOUT' if rec.get('timed_out') else ''}{' ERR' if rec.get('error') else ''}",
              flush=True)
        return rec

    print(f"ARC-AGI distill: scope={args.scope} {len(rows)} tasks | model={args.model} "
          f"parallel={args.parallel} max_turns={args.max_iterations} timeout={args.task_timeout}s")

    # Phase 1: solve every not-yet-attempted task.
    todo, skipped = [], 0
    for r in rows:
        if _existing(args, r["task_id"], "solve") is not None and not args.force:
            skipped += 1
        else:
            todo.append((r["task_id"], "solve", None))
    print(f"Phase 1 (solve): {len(todo)} to attempt, {skipped} already done")
    if todo:
        _run_pool(args, todo, _emit)

    # Phase 2: refine each solved task.
    if args.refine:
        rjobs = []
        for r in rows:
            rec = _existing(args, r["task_id"], "solve")
            if not (rec and rec.get("grade", {}).get("fully_correct") and rec.get("final_solution")):
                continue
            if _existing(args, r["task_id"], "refine") is not None and not args.force:
                continue
            g = rec["grade"]
            rjobs.append((r["task_id"], "refine", {
                "task_id": r["task_id"],
                "prior_solution": rec["final_solution"],
                "prior_size": g.get("size"),
                "raw_baseline": g.get("raw_baseline"),
            }))
        print(f"Phase 2 (refine): {len(rjobs)} solved tasks to refine")
        if rjobs:
            _run_pool(args, rjobs, _emit)

    out_f.close()
    n_solve = sum(1 for _, m, _, _ in summary if m == "solve")
    n_ok = sum(1 for _, m, c, _ in summary if m == "solve" and c)
    n_win = sum(1 for _, _, c, rw in summary if c and rw and rw > 0)
    print(f"\n=== this run: solve {n_ok}/{n_solve} fully correct; {n_win} compression wins ===")
    print(f"combined -> {results_path}")
    print(f"per-task -> {args.out}/<ver__id>/<solve|refine>/")
    return 0


def main() -> int:
    args = _build_argparser().parse_args()
    if "SFDC_GATEWAY_KEY" not in os.environ:
        sys.exit("set SFDC_GATEWAY_KEY")
    if not Path(args.grader_py).exists():
        sys.exit(f"grader python not found: {args.grader_py}")
    return _run_worker(args) if args.single else _run_orchestrator(args)


# Helper script (runs in the grader venv) to scaffold a task to baseline, render its
# grids as plain text for the prompt, and compute the raw-array baseline object count.
_SCAFFOLD_HELPER = r'''
import sys, json
sys.path.insert(0, ".")
import arckit, hy
import eval as E

ver, tid = sys.argv[1], sys.argv[2]
task = arckit.load_single(tid, version=ver)
pairs = list(task.train) + list(task.test)

# Baseline scaffold: literal input grids + stub to-output (eval.py's scaffold form).
in_forms = [E.grid_to_hy(inp) for inp, out in pairs]
src = [f";; tasks/{ver}/{E.split_for(ver, tid)}/{tid}.hy — {len(pairs)} input/output pairs",
       "(import library *)", "(import hy)", "", "(defn inputs []", "  ["]
for e in in_forms:
    src.append(f"    (quote {e})")
src += ["  ])", "",
        '(defn to-output [expr]',
        '  "Stub: must be replaced with a real structural rewrite of expr."',
        "  expr)", ""]
scaffold = "\n".join(src)
# Write it to the on-disk task path so the agent starts from baseline.
E.task_path(ver, tid).write_text(scaffold)

# Raw-array baseline = literal input cells + literal output cells.
out_forms = [E.grid_to_hy(out) for inp, out in pairs]
in_objs = E.count_objects(list(hy.read_many("\n".join(str(f) for f in in_forms))))
out_objs = E.count_objects(list(hy.read_many("\n".join(str(f) for f in out_forms))))

# Plain-text grids for the prompt.
def render(g):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)
parts = []
for i, (inp, out) in enumerate(pairs):
    tag = "Test" if i >= len(task.train) else "Train"
    parts.append(f"### {tag} pair {i}\nInput:\n```\n{render(inp)}\n```\nOutput:\n```\n{render(out)}\n```")
grids = "\n\n".join(parts)

print("@@DESC@@" + json.dumps({
    "n_pairs": len(pairs), "raw_baseline": int(in_objs + out_objs),
    "scaffold": scaffold, "grids": grids,
}))
'''


if __name__ == "__main__":
    raise SystemExit(main())
