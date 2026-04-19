"""Helpers for the curated 10-task SWE-Gym sample."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_PAGE_SIZE = 50
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "polar" / "swegym_sample_10.json"
RUNTIME_IMAGE_PREFIX = "polar-swegym-runtime"

# Curated from the 293-row training split to cover multiple repositories while
# staying on plain SWE-Gym text instances (no multimodal/image_assets rows).
SAMPLE_TASKS: list[dict[str, str]] = [
    {
        "instance_id": "getmoto__moto-7365",
        "repo": "getmoto/moto",
        "summary": "Fix Decimal arithmetic in mocked DynamoDB update_item ADD handling.",
    },
    {
        "instance_id": "python__mypy-10392",
        "repo": "python/mypy",
        "summary": "Search @python2 subdirectories when resolving Python 2 typeshed paths.",
    },
    {
        "instance_id": "conan-io__conan-13721",
        "repo": "conan-io/conan",
        "summary": "Add profile_name as a variable during profile rendering.",
    },
    {
        "instance_id": "iterative__dvc-1809",
        "repo": "iterative/dvc",
        "summary": "Infer metrics type automatically from file suffixes like .json.",
    },
    {
        "instance_id": "dask__dask-10441",
        "repo": "dask/dask",
        "summary": "Fix invalid append-mode handling in DataFrame to_csv.",
    },
    {
        "instance_id": "pydantic__pydantic-8072",
        "repo": "pydantic/pydantic",
        "summary": "Remove the __pydantic_self__ constructor edge case.",
    },
    {
        "instance_id": "pandas-dev__pandas-58335",
        "repo": "pandas-dev/pandas",
        "summary": "Avoid incorrect duplicate-column warning for to_dict orient=tight.",
    },
    {
        "instance_id": "facebookresearch__hydra-1783",
        "repo": "facebookresearch/hydra",
        "summary": "Improve the missing-default error for non-standard package layouts.",
    },
    {
        "instance_id": "bokeh__bokeh-13636",
        "repo": "bokeh/bokeh",
        "summary": "Use globally unique, CSS-safe JSON script IDs in embeddings.",
    },
    {
        "instance_id": "Project-MONAI__MONAI-2238",
        "repo": "Project-MONAI/MONAI",
        "summary": "Prevent CopyItemsd from writing copied data back into the same key.",
    },
    # --- Expanded sample (20 extra tasks) for richer training signal ---
    {
        "instance_id": "getmoto__moto-4950",
        "repo": "getmoto/moto",
        "summary": "Fix Timestream Write response payload shape.",
    },
    {
        "instance_id": "getmoto__moto-6178",
        "repo": "getmoto/moto",
        "summary": "Handle brackets in DynamoDB KeyConditionExpression.",
    },
    {
        "instance_id": "getmoto__moto-4986",
        "repo": "getmoto/moto",
        "summary": "Return encryptionConfig as a list from EKS CreateCluster.",
    },
    {
        "instance_id": "pandas-dev__pandas-52076",
        "repo": "pandas-dev/pandas",
        "summary": "Accept pyarrow string arrays with explicit dtype in Series.",
    },
    {
        "instance_id": "pandas-dev__pandas-57758",
        "repo": "pandas-dev/pandas",
        "summary": "Support Boolean columns in the DataFrame Interchange Protocol.",
    },
    {
        "instance_id": "pandas-dev__pandas-50988",
        "repo": "pandas-dev/pandas",
        "summary": "Allow numexpr query() with column names like 'max'/'min'.",
    },
    {
        "instance_id": "Project-MONAI__MONAI-4800",
        "repo": "Project-MONAI/MONAI",
        "summary": "Fix flake8-pyi warnings reported in MONAI stubs.",
    },
    {
        "instance_id": "Project-MONAI__MONAI-7548",
        "repo": "Project-MONAI/MONAI",
        "summary": "Correct medicalnet_resnet50 typo in perceptual loss.",
    },
    {
        "instance_id": "dask__dask-8484",
        "repo": "dask/dask",
        "summary": "Make nanmin/nanmax work on scalar inputs.",
    },
    {
        "instance_id": "dask__dask-8686",
        "repo": "dask/dask",
        "summary": "Match NumPy semantics for concatenate with axis=None.",
    },
    {
        "instance_id": "python__mypy-15184",
        "repo": "python/mypy",
        "summary": "Use fully qualified names in assert_type when ambiguous.",
    },
    {
        "instance_id": "python__mypy-10036",
        "repo": "python/mypy",
        "summary": "Avoid daemon crash when a stub package is removed.",
    },
    {
        "instance_id": "iterative__dvc-4613",
        "repo": "iterative/dvc",
        "summary": "Print current cache dir when `dvc cache dir` has no flags.",
    },
    {
        "instance_id": "iterative__dvc-4785",
        "repo": "iterative/dvc",
        "summary": "Respect HTTP status codes when using HTTP(S) remotes.",
    },
    {
        "instance_id": "conan-io__conan-9431",
        "repo": "conan-io/conan",
        "summary": "Allow c++17 profiles to build with gcc 5.",
    },
    {
        "instance_id": "conan-io__conan-14164",
        "repo": "conan-io/conan",
        "summary": "Handle symlinked home directories in Conan cache paths.",
    },
    {
        "instance_id": "pydantic__pydantic-8965",
        "repo": "pydantic/pydantic",
        "summary": "Thread a context object through model serialization methods.",
    },
    {
        "instance_id": "pydantic__pydantic-8316",
        "repo": "pydantic/pydantic",
        "summary": "Fix to_snake alias generator edge cases.",
    },
    {
        "instance_id": "facebookresearch__hydra-1915",
        "repo": "facebookresearch/hydra",
        "summary": "Support callables targeting nested classes in instantiate.",
    },
    {
        "instance_id": "modin-project__modin-7225",
        "repo": "modin-project/modin",
        "summary": "Stop modin.pandas.api.extensions from clobbering pandas re-exports.",
    },
]


def sample_instance_ids() -> list[str]:
    return [item["instance_id"] for item in SAMPLE_TASKS]


def sanitize_instance_id(instance_id: str) -> str:
    normalized = instance_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized.replace("__", "--"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def base_image_for_instance_id(instance_id: str) -> str:
    suffix = instance_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def derived_runtime_image(instance_id: str) -> str:
    return f"{RUNTIME_IMAGE_PREFIX}:{sanitize_instance_id(instance_id)}"


def sample_rows_url(offset: int, length: int) -> str:
    return (
        f"{DATASET_ROWS_URL}?dataset={DATASET_NAME}"
        f"&config={DATASET_CONFIG}&split={DATASET_SPLIT}"
        f"&offset={offset}&length={length}"
    )


def fetch_sample_instances(
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    cache_file = cache_path or DEFAULT_CACHE_PATH
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text())
        # Invalidate cache when SAMPLE_TASKS changes (e.g. list grew from 10 to 30).
        if {item.get("instance_id") for item in cached} == set(sample_instance_ids()):
            return cached

    wanted = set(sample_instance_ids())
    found: dict[str, dict[str, Any]] = {}
    offset = 0

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        while len(found) < len(wanted):
            response = client.get(sample_rows_url(offset, DATASET_PAGE_SIZE))
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("rows") or []
            if not rows:
                break
            for row in rows:
                instance = (row.get("row") or {}).get("instance") or {}
                if not isinstance(instance, dict):
                    continue
                instance_id = instance.get("instance_id")
                if instance_id in wanted and instance_id not in found:
                    found[instance_id] = instance
            offset += len(rows)

    missing = [instance_id for instance_id in sample_instance_ids() if instance_id not in found]
    if missing:
        raise RuntimeError(
            "Failed to fetch all curated SWE-Gym sample tasks. "
            f"Missing instance_ids: {missing}"
        )

    ordered = [found[instance_id] for instance_id in sample_instance_ids()]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(ordered, indent=2, ensure_ascii=True, sort_keys=True))
    return ordered
