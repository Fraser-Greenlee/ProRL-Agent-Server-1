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
    ap.add_argument("--source", choices=["arcagi", "rl_tasks"], default="arcagi",
                    help="arcagi = difficulty-ordered arckit tasks; "
                         "rl_tasks = synthetic easy tasks under sft/rl_tasks/")
    args = ap.parse_args()

    repo = auto_compress_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "baseline_run"))

    import hy  # noqa: F401  (registers the Hy import hook for eval.py)
    from eval import count_objects

    if args.source == "rl_tasks":
        rows = _build_rl_tasks_rows(repo, args.n_tasks, count_objects, hy)
    else:
        rows = _build_arcagi_rows(repo, args.n_tasks, count_objects, hy)

    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n"
    )
    print(f"Wrote {len(rows)} tasks to {args.output}")
    print(f"  baseline sizes: min={min(r['metadata']['baseline_size'] for r in rows)} "
          f"max={max(r['metadata']['baseline_size'] for r in rows)}")


def _build_arcagi_rows(repo, n_tasks, count_objects, hy):
    from baseline_run.prompt import build as build_prompt
    from baseline_run.run import scaffold_baseline
    task_ids = load_task_ids(repo, n_tasks)
    rows = []
    for tid in task_ids:
        baseline_body = scaffold_baseline(tid)
        baseline_size = count_objects(list(hy.read_many(baseline_body)))
        instruction = build_prompt(tid, baseline_body, repo_root=repo)
        rows.append({
            "prompt": [{"role": "user", "content": instruction}],
            "label": "",
            "metadata": {"task_id": tid, "baseline_size": baseline_size,
                         "baseline_body": baseline_body, "task_kind": "arcagi",
                         "split": "train"},
        })
    return rows


def _build_rl_tasks_rows(repo, n_tasks, count_objects, hy):
    """One row per synthetic task in sft/rl_tasks/.  Baseline = raw-grid
    scaffold from pairs.json (score_synthetic.scaffold)."""
    import json as _json
    import numpy as np
    sys.path.insert(0, str(repo / "sft" / "rl_tasks"))
    from score_synthetic import baseline_body as synth_baseline_body  # noqa: E402

    rl_dir = repo / "sft" / "rl_tasks"
    library_body = (repo / "library.hy").read_text()
    task_ids = sorted(
        d.name for d in rl_dir.iterdir()
        if d.is_dir() and (d / "pairs.json").exists()
    )
    if n_tasks and n_tasks > 0:
        task_ids = task_ids[:n_tasks]

    rows = []
    for tid in task_ids:
        # In-memory baseline (does NOT touch the on-disk reference .hy; the
        # image build resets the .hy files separately).
        baseline_body = synth_baseline_body(rl_dir, tid)
        baseline_size = count_objects(list(hy.read_many(baseline_body)))
        pairs = _json.loads((rl_dir / tid / "pairs.json").read_text())
        instruction = _synth_prompt(tid, pairs, baseline_body, library_body, np)
        rows.append({
            "prompt": [{"role": "user", "content": instruction}],
            "label": "",
            "metadata": {"task_id": tid, "baseline_size": baseline_size,
                         "baseline_body": baseline_body, "task_kind": "synthetic",
                         "split": "train"},
        })
    return rows


def _render_grid(grid) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def _synth_prompt(task_id, pairs, baseline_body, library_body, np) -> str:
    blocks = []
    for i, p in enumerate(pairs):
        blocks.append(f"### Pair {i}\nInput:\n```\n{_render_grid(p['input'])}\n```\n"
                      f"Output:\n```\n{_render_grid(p['output'])}\n```\n")
    grids = "\n".join(blocks).rstrip()
    return _SYNTH_TEMPLATE.format(
        task_id=task_id, grids=grids, n_pairs=len(pairs),
        baseline_body=baseline_body, library_body=library_body,
    )


_SYNTH_TEMPLATE = """\
You are compressing one ARC-style task into a structural Hy program.

# Goal

Edit `sft/rl_tasks/{task_id}/{task_id}.hy` so it is SMALLER (fewer objects under
the count_objects metric) while every input/output pair still matches the ground
truth.

# Rules

- Only edit `sft/rl_tasks/{task_id}/{task_id}.hy`. All other files are read-only.
- Keep two functions: `(defn inputs [] ...)` and `(defn to-output [expr] ...)`.
  `inputs()[i]` must evaluate to input i; `(to-output (inputs)[i])` must evaluate
  to output i (ARC 0..9 palette).
- If you use `->`, `cond`, `when`, `for`, or other hyrule macros, the file must
  start with `(require hyrule * :readers *)` — they are NOT re-exported by
  `(require library *)`.
- Write `inputs()[i]` as a CONSTRUCTIVE expression and `to-output` as a
  STRUCTURAL rewrite, using primitives from `library.hy` below.

# Verify your work (do this — don't guess)

After every edit, run this exact command from the terminal to check correctness
and see your compression score:

```
python /opt/auto-compress/sft/rl_tasks/score_synthetic.py \\
  --root /opt/auto-compress/sft/rl_tasks --verify {task_id}
```

It prints, per pair, whether the output is correct and — when it's wrong — a
grid DELTA marking each off cell as `[got!=want]`, so you can see exactly what
to fix. It also prints `size`, the `baseline`, and the `reward`. Iterate edit →
verify until it reports `{n_pairs}/{n_pairs} correct`, then keep shrinking the
program (fewer objects) while it stays fully correct. Do NOT try to import or
`hy.eval` the file yourself — use this command.

# Task {task_id}

## Grids

{grids}

## Current body of `{task_id}.hy`

```hy
{baseline_body}
```

## `library.hy` (shared primitives — read-only)

```hy
{library_body}
```

Produce a smaller `.hy` body that still passes. Verify with the command above
before finishing.
"""


if __name__ == "__main__":
    main()
