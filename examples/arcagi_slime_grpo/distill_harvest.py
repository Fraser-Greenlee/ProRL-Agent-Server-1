#!/usr/bin/env python3
"""Harvest distillation trajectories into an SFT dataset.

Scans the per-task artifacts produced by distill_generate.py
(<out>/<task_id>/<solve|refine>/trajectory.json) and emits one SFT example per
KEPT trajectory: the full replayable conversation (system + user + assistant
turns with chain-of-thought + tool calls + tool results), tagged with the grade.

Selection (one attempt per task, but solve AND refine are separate trajectories):
  - default keep: fully_correct (the agent produced a valid solution).
  - --min-reward R: additionally require reward >= R (e.g. 0.0 keeps correct-but-
    uncompressed; >0 keeps only compression wins).
  - --best-per-task: among kept trajectories for the same task_id, keep only the
    one with the highest reward (so a refine that beat the solve wins; ties → fewer
    turns). Without it, every kept trajectory is emitted (solve + refine both).

Output: a jsonl where each line is
  {task_id, mode, reward, compression_pct, n_turns, messages: [...]}
`messages` is the OpenAI-style conversation as captured (assistant turns retain
`reasoning_content` + `thinking_blocks`). Converting to a specific student chat
template (e.g. Qwen <think> tags) is a deliberate downstream step — see
--strip-thinking / DISTILL_README. We DON'T bake a template in here so the same
harvest can target different SFT recipes.

Usage:
  python distill_harvest.py --in runs/distill1 --out runs/distill1/sft.jsonl
  python distill_harvest.py --in runs/distill1 --out sft.jsonl --min-reward 0.0001 --best-per-task
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_trajectories(in_dir: Path):
    """Yield (task_id, mode, record) for every trajectory.json under in_dir."""
    for traj in sorted(in_dir.glob("*/*/trajectory.json")):
        mode = traj.parent.name           # solve | refine
        task_id = traj.parent.parent.name
        try:
            rec = json.loads(traj.read_text())
        except Exception:
            continue
        yield task_id, mode, rec


def _strip_thinking(messages: list) -> list:
    """Drop chain-of-thought from assistant turns (keep tool_calls + content).
    Use when the SFT recipe trains on actions only, or to shrink long trajectories
    under the student's context budget."""
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            m = {k: v for k, v in m.items()
                 if k not in ("reasoning_content", "thinking_blocks")}
        out.append(m)
    return out


def _n_turns(messages: list) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True,
                    help="distill_generate.py --out directory")
    ap.add_argument("--out", type=Path, required=True, help="SFT jsonl to write")
    ap.add_argument("--min-reward", type=float, default=None,
                    help="require reward >= this (omit = keep any fully_correct)")
    ap.add_argument("--require-correct", action="store_true", default=True,
                    help="require fully_correct (default on)")
    ap.add_argument("--keep-incorrect", dest="require_correct", action="store_false",
                    help="also keep not-fully-correct trajectories")
    ap.add_argument("--best-per-task", action="store_true",
                    help="emit only the highest-reward kept trajectory per task_id")
    ap.add_argument("--strip-thinking", action="store_true",
                    help="drop CoT (reasoning_content/thinking_blocks) from assistant turns")
    args = ap.parse_args()

    if not args.in_dir.is_dir():
        raise SystemExit(f"--in not a directory: {args.in_dir}")

    kept: list[dict] = []
    stats = {"total": 0, "correct": 0, "wins": 0, "kept": 0}
    for task_id, mode, rec in _iter_trajectories(args.in_dir):
        stats["total"] += 1
        g = rec.get("grade") or {}
        correct = bool(g.get("fully_correct"))
        reward = g.get("reward")
        if correct:
            stats["correct"] += 1
        if reward is not None and reward > 0:
            stats["wins"] += 1

        if args.require_correct and not correct:
            continue
        if args.min_reward is not None and not (reward is not None and reward >= args.min_reward):
            continue

        msgs = rec.get("messages") or []
        if not msgs:
            continue
        if args.strip_thinking:
            msgs = _strip_thinking(msgs)

        kept.append({
            "task_id": task_id,
            "mode": mode,
            "reward": reward,
            "compression_pct": g.get("compression_pct"),
            "n_turns": _n_turns(msgs),
            "model": rec.get("model"),
            "messages": msgs,
        })

    if args.best_per_task:
        best: dict[str, dict] = {}
        for ex in kept:
            cur = best.get(ex["task_id"])
            r = ex["reward"] if ex["reward"] is not None else -1e9
            cr = cur["reward"] if (cur and cur["reward"] is not None) else -1e9
            # higher reward wins; tie → fewer turns (cleaner trajectory)
            if cur is None or r > cr or (r == cr and ex["n_turns"] < cur["n_turns"]):
                best[ex["task_id"]] = ex
        kept = list(best.values())

    kept.sort(key=lambda e: (e["task_id"], e["mode"]))
    stats["kept"] = len(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    n_tasks = len({e["task_id"] for e in kept})
    print(f"scanned {stats['total']} trajectories: "
          f"{stats['correct']} fully-correct, {stats['wins']} reward>0")
    print(f"kept {stats['kept']} SFT examples covering {n_tasks} distinct task(s)"
          f"{' (best-per-task)' if args.best_per_task else ''}"
          f"{' [thinking stripped]' if args.strip_thinking else ''}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
