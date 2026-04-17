#!/usr/bin/env python3
"""Prepare the SWE-Gym 10-task sample as a JSONL dataset for Slime training.

Each row contains:
  - prompt:       the problem_statement
  - label:        always "" (reward comes from Polar evaluator, not label matching)
  - metadata:     instance dict + polar_image tag (used by polar_config.yaml template)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_tasks import (
    derived_image_for_harness,
    fetch_sample_instances,
)

OUTPUT = Path(__file__).resolve().parent / "swegym_10_tasks.jsonl"


def main() -> None:
    instances = fetch_sample_instances()
    rows: list[dict] = []
    for instance in instances:
        instance_id = str(instance["instance_id"])
        image_tag = derived_image_for_harness(instance_id, "swe_agent")
        rows.append({
            "prompt": instance["problem_statement"].strip(),
            "label": "",
            "metadata": {
                "instance_id": instance_id,
                "instance": instance,
                "polar_image": f"docker-daemon:{image_tag}",
            },
        })
    OUTPUT.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")
    print(f"Wrote {len(rows)} tasks to {OUTPUT}")


if __name__ == "__main__":
    main()
