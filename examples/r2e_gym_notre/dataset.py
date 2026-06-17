"""Load and cache the R2E-Gym-Subset dataset.

Each instance ships a ready-to-run Docker image (``namanjain12/...`` on Docker
Hub) containing the repo checked out at the buggy commit under ``/testbed`` plus
the gold test suite under ``/r2e_tests``. Rollouts pull the image directly; the
agent edits ``/testbed`` and is graded by re-running ``r2e_tests``.

Dataset: https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset
Environment: https://github.com/R2E-Gym/R2E-Gym
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
DATASET_SPLIT = "train"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "polar" / "r2e_gym_subset.json"

# Canonical R2E-Gym test command: the official images expose the gold suite at
# /r2e_tests (symlinked into /testbed by the prepare step).
DEFAULT_TEST_COMMAND = "python -m pytest r2e_tests -rA --tb=short"
REPO_DIR = "/testbed"


def sanitize_instance_id(instance_id: str) -> str:
    normalized = instance_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized.replace("__", "--"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def instance_id_for(instance: dict[str, Any]) -> str:
    explicit = instance.get("instance_id") or instance.get("id")
    if explicit:
        return str(explicit)
    repo = str(instance.get("repo_name") or "repo")
    commit = str(instance.get("commit_hash") or "")
    return f"{repo}-{commit[:12]}".strip("-")


def docker_image_for(instance: dict[str, Any]) -> str:
    image = instance.get("docker_image") or instance.get("runtime_image") or instance.get("image")
    if not image:
        raise ValueError(f"R2E instance {instance_id_for(instance)} has no docker_image")
    return str(image)


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    instance["instance_id"] = instance_id_for(instance)
    instance.setdefault("test_command", DEFAULT_TEST_COMMAND)
    # expected_output_json, if present, lets the evaluator match per-test gold
    # outcomes; otherwise it falls back to the pytest exit code.
    value = instance.get("expected_output_json")
    if isinstance(value, str):
        try:
            instance["expected_output_json"] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            instance.pop("expected_output_json", None)
    return instance


def load_r2e_gym_subset(
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load R2E-Gym-Subset via HuggingFace ``datasets``, with a local JSON cache."""
    cache_file = cache_path or DEFAULT_CACHE_PATH
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    instances = [_normalize_instance(dict(row)) for row in ds]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(instances, indent=2, ensure_ascii=True, sort_keys=True))
    print(f"Cached {len(instances)} R2E-Gym-Subset instances to {cache_file}")
    return instances
