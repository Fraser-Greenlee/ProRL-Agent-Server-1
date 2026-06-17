"""Load and cache the LiteResearcher-Data question/answer dataset.

Each item is a research question with one or more reference answers. The agent
answers it using the `search`/`visit` MCP tools (backed by a live LiteResearcher
retrieval service) and the `answer_judge` evaluator matches the final
`<answer>...</answer>` against the references.

Dataset: https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data
Environment: https://github.com/simplexai-labs/LiteResearcher
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_NAME = "simplex-ai-inc/LiteResearcher-Data"
DEFAULT_STAGE = "stage1"


def cache_path_for(stage: str) -> Path:
    return Path.home() / ".cache" / "polar" / f"literesearcher_{stage}.json"


def question_id(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def _reference(row: dict[str, Any]) -> list[str] | str:
    direct = row.get("answer") or row.get("ground_truth")
    if isinstance(direct, list):
        values = [str(v).strip() for v in direct if str(v).strip()]
        if values:
            return values
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    reward_model = row.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except (json.JSONDecodeError, TypeError):
            reward_model = None
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            target = ground_truth.get("target")
            if isinstance(target, list):
                values = [str(v).strip() for v in target if str(v).strip()]
                if values:
                    return values
            if isinstance(target, str) and target.strip():
                return target.strip()
    return ""


def _mask_url(row: dict[str, Any]) -> str | None:
    extra = row.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra = None
    if not isinstance(extra, dict):
        return None
    url = extra.get("mask_url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    tools_kwargs = extra.get("tools_kwargs")
    if isinstance(tools_kwargs, dict):
        for tool in ("browse", "search"):
            entry = tools_kwargs.get(tool)
            if isinstance(entry, dict):
                create_kwargs = entry.get("create_kwargs")
                if isinstance(create_kwargs, dict):
                    url = create_kwargs.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()
    return None


def load_literesearcher_data(
    *,
    stage: str = DEFAULT_STAGE,
    refresh: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load LiteResearcher-Data (a stage's ``train.parquet``) with a JSON cache."""
    cache_file = cache_path or cache_path_for(stage)
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, data_files=f"{stage}/train.parquet", split="train")
    items: list[dict[str, Any]] = []
    for row in ds:
        row = dict(row)
        question = str(row.get("question") or row.get("prompt") or "").strip()
        reference = _reference(row)
        if not question or not reference:
            continue
        items.append(
            {
                "question": question,
                "answer": reference,
                "data_source": row.get("data_source"),
                "mask_url": _mask_url(row),
                "source_stage": stage,
            }
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(items, indent=2, ensure_ascii=True, sort_keys=True))
    print(f"Cached {len(items)} LiteResearcher {stage} item(s) to {cache_file}")
    return items
