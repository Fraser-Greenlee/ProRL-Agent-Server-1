#!/usr/bin/env python3
"""Build the arcagi train JSONL for Slime + Polar GRPO.

Picks the first N task IDs from the auto-compress difficulty TSV (easy → hard;
the same ordering the baseline run used), and for each emits one row:

  - prompt:   chat-formatted [{"role": "user", "content": <full instruction>}]
              where <instruction> is exactly what baseline_run/prompt.py renders
              (task grids + LARC sidecar + arc-dsl reference solver + the
              baseline scaffold body + the full library.hy).
  - label:    "" (reward comes from the Polar arc_compress evaluator).
  - metadata: task_id, baseline_size (object count of the scaffold), split,
              and the baseline_body (so the evaluator / topology can reconstruct
              the starting file without re-deriving it).

Run this inside the auto-compress environment (needs arckit + eval.py +
library.hy on the path), e.g.:

    AUTO_COMPRESS=/path/to/auto-compress \
      python examples/arcagi_slime_grpo/prepare_data.py --n-tasks 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

DEFAULT_N_TASKS = 40
OUTPUT = Path(__file__).resolve().parent / "arcagi_train.jsonl"


def auto_compress_root() -> Path:
    root = os.environ.get("AUTO_COMPRESS")
    if root:
        p = Path(root).expanduser().resolve()
        if not (p / "eval.py").exists():
            sys.exit(f"AUTO_COMPRESS={p} has no eval.py")
        return p
    # Fall back to the sibling checkout layout used on dev machines.
    for cand in (
        Path.home() / "Projects" / "auto-compress",
        Path("/home/fraser_convergence_ai/auto-compress"),
    ):
        if (cand / "eval.py").exists():
            return cand
    sys.exit("Set AUTO_COMPRESS to the auto-compress repo root.")


def load_task_ids(repo: Path, n_tasks: int) -> list[str]:
    tsv = repo / "compression-difficulty.tsv"
    if not tsv.exists():
        sys.exit(f"difficulty TSV not found at {tsv}")
    ids: list[str] = []
    with tsv.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids.append(row["task_id"])
    return ids[:n_tasks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    repo = auto_compress_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "baseline_run"))

    # Imported from the auto-compress repo (need its arckit + library.hy).
    import hy  # noqa: F401  (registers the Hy import hook for eval.py)
    from eval import count_objects
    from baseline_run.prompt import build as build_prompt
    from baseline_run.run import scaffold_baseline

    task_ids = load_task_ids(repo, args.n_tasks)
    rows: list[dict] = []
    for tid in task_ids:
        baseline_body = scaffold_baseline(tid)
        baseline_size = count_objects(list(hy.read_many(baseline_body)))
        instruction = build_prompt(tid, baseline_body, repo_root=repo)
        rows.append(
            {
                "prompt": [{"role": "user", "content": instruction}],
                "label": "",
                "metadata": {
                    "task_id": tid,
                    "baseline_size": baseline_size,
                    "baseline_body": baseline_body,
                    "split": "train",
                },
            }
        )

    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n"
    )
    print(f"Wrote {len(rows)} tasks to {args.output}")
    print(f"  baseline sizes: min={min(r['metadata']['baseline_size'] for r in rows)} "
          f"max={max(r['metadata']['baseline_size'] for r in rows)}")


if __name__ == "__main__":
    main()
