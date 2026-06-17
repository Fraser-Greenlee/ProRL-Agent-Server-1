#!/usr/bin/env python3
"""Prepare CUA-Gym web tasks: clone the app sources and materialize task bundles.

Steps:
  1. Clone CUA-Gym-Hub (the mock web apps) if not already present.
  2. Download ``data/tasks.parquet`` + the task-bundle archive from the
     ``xlangai/CUA-Gym`` HuggingFace dataset.
  3. Filter to runnable web tasks (``platform=web``, ``app_family=mock_web``,
     ``setup_kind=py``) whose app exists in the cloned hub.
  4. Extract each task's ``task.json`` / ``initial_setup.py`` / ``reward.py`` and
     materialize the ``__CUA_GYM_*_URL__`` placeholders to a per-task local port
     (and pin the session id), writing ``data/cua_web_tasks.jsonl``.

Dataset: https://huggingface.co/datasets/xlangai/CUA-Gym
Apps:    https://github.com/xlang-ai/CUA-Gym-Hub
Env:     https://github.com/xlang-ai/CUA-Gym

    python prepare_data.py --max-tasks 10
    python prepare_data.py --max-tasks 50 --difficulty easy,medium --start-port 18100
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
CUA_REPO = "xlangai/CUA-Gym"
TASKS_PARQUET = "data/tasks.parquet"
TASKS_ARCHIVE = "artifacts/cua_gym_tasks_v1.tar.zst"
HUB_GIT_URL = "https://github.com/xlang-ai/CUA-Gym-Hub.git"

DEFAULT_OUTPUT = EXAMPLE_DIR / "data" / "cua_web_tasks.jsonl"
DEFAULT_HUB_DIR = EXAMPLE_DIR / "data" / "cua_gym_hub"
DEFAULT_MATERIALIZED_ROOT = EXAMPLE_DIR / "data" / "materialized_tasks"
PLACEHOLDER_RE = re.compile(r"__CUA_GYM_[A-Z0-9_]+_(?:URL|HOST)__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--difficulty", default="easy,medium")
    parser.add_argument("--start-port", type=int, default=18100)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hub-dir", default=str(DEFAULT_HUB_DIR))
    parser.add_argument("--materialized-root", default=str(DEFAULT_MATERIALIZED_ROOT))
    parser.add_argument("--refresh-hub", action="store_true", help="Re-clone CUA-Gym-Hub.")
    return parser.parse_args()


def clone_hub(hub_dir: Path, *, refresh: bool) -> Path:
    if refresh and hub_dir.exists():
        shutil.rmtree(hub_dir)
    if not hub_dir.exists():
        hub_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {HUB_GIT_URL} -> {hub_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", HUB_GIT_URL, str(hub_dir)],
            check=True,
        )
    websites = hub_dir / "websites"
    if not websites.is_dir():
        raise SystemExit(f"Expected {websites} after cloning CUA-Gym-Hub")
    return websites


def load_parquet_rows(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def extract_bundles(*, archive_path: str, wanted: set[str], output_root: Path) -> None:
    """Stream-decompress the .tar.zst via the zstd CLI and extract wanted task dirs."""
    if not shutil.which("zstd"):
        raise SystemExit("The 'zstd' CLI is required to unpack the CUA task archive (apt install zstd).")
    remaining = set(wanted)
    proc = subprocess.Popen(["zstd", "-dc", archive_path], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                name = member.name
                if name.startswith("._") or "/._" in name:
                    continue
                top = name.split("/", 1)[0]
                if top not in remaining:
                    continue
                _safe_extract(tar, member, output_root)
                if (output_root / top / "reward.py").exists():
                    remaining.discard(top)
                    if not remaining:
                        break
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def _safe_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, output_root: Path) -> None:
    target = (output_root / member.name).resolve()
    if not str(target).startswith(str(output_root.resolve())):
        raise RuntimeError(f"Unsafe CUA archive path: {member.name}")
    tar.extract(member, output_root)


def collect_placeholders(paths: list[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        values.update(PLACEHOLDER_RE.findall(path.read_text(encoding="utf-8")))
    return values


def materialize_bundle(*, target_dir: Path, base_url: str, sid: str) -> None:
    host = base_url.removeprefix("http://").removeprefix("https://")
    for path in target_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for placeholder in set(PLACEHOLDER_RE.findall(text)):
            replacement = host if placeholder.endswith("_HOST__") else base_url
            text = text.replace(placeholder, replacement)
        text = re.sub(r"sid\s*=\s*str\(uuid\.uuid4\(\)\)", f"sid = {sid!r}", text)
        path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    websites = clone_hub(Path(args.hub_dir).resolve(), refresh=args.refresh_hub)
    app_dirs = {p.name for p in websites.iterdir() if p.is_dir()}
    allowed_difficulty = {v.strip() for v in args.difficulty.split(",") if v.strip()}

    parquet_path = hf_hub_download(repo_id=CUA_REPO, filename=TASKS_PARQUET, repo_type="dataset")
    archive_path = hf_hub_download(repo_id=CUA_REPO, filename=TASKS_ARCHIVE, repo_type="dataset")
    rows = load_parquet_rows(parquet_path)

    def is_web_task(row: dict[str, Any], *, with_difficulty: bool) -> bool:
        ok = (
            row.get("platform") == "web"
            and row.get("app_family") == "mock_web"
            and row.get("setup_kind") == "py"
            and row.get("app_type") in app_dirs
        )
        if with_difficulty and allowed_difficulty:
            ok = ok and row.get("difficulty") in allowed_difficulty
        return ok

    candidates = [r for r in rows if is_web_task(r, with_difficulty=True)][: args.candidate_count]
    if len(candidates) < args.max_tasks:  # relax difficulty if too few
        candidates = [r for r in rows if is_web_task(r, with_difficulty=False)][: args.candidate_count]

    materialized_root = Path(args.materialized_root).resolve()
    if materialized_root.exists():
        shutil.rmtree(materialized_root)
    materialized_root.mkdir(parents=True, exist_ok=True)

    raw_root = materialized_root / "_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    wanted = {str(r["archive_member"]) for r in candidates}
    extract_bundles(archive_path=archive_path, wanted=wanted, output_root=raw_root)

    prepared: list[dict[str, Any]] = []
    for row in candidates:
        if len(prepared) >= args.max_tasks:
            break
        task_id = str(row["id"])
        source_dir = raw_root / str(row["archive_member"])
        task_json = source_dir / "task.json"
        setup_py = source_dir / "initial_setup.py"
        reward_py = source_dir / "reward.py"
        if not (task_json.exists() and setup_py.exists() and reward_py.exists()):
            continue
        placeholders = collect_placeholders([setup_py, reward_py])
        url_placeholders = {p for p in placeholders if p.endswith("_URL__")}
        if len(url_placeholders) != 1:  # single-mock web tasks only
            continue

        port = int(args.start_port) + len(prepared)
        sid = task_id
        base_url = f"http://127.0.0.1:{port}"
        target_dir = materialized_root / task_id
        shutil.copytree(source_dir, target_dir)
        materialize_bundle(target_dir=target_dir, base_url=base_url, sid=sid)

        task = json.loads((target_dir / "task.json").read_text(encoding="utf-8"))
        task.update(
            {
                "id": task_id,
                "sid": sid,
                "port": port,
                "base_url": base_url,
                "bundle_dir": str(target_dir),
                "app_source": str(websites / str(row["app_type"])),
                "app_type": row.get("app_type"),
                "difficulty": row.get("difficulty"),
                "source_dataset": CUA_REPO,
            }
        )
        prepared.append(task)

    shutil.rmtree(raw_root, ignore_errors=True)
    if not prepared:
        raise SystemExit("No usable CUA web tasks prepared. Try raising --candidate-count.")
    if len(prepared) < args.max_tasks:
        print(f"Warning: prepared {len(prepared)} task(s); requested {args.max_tasks}.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in prepared:
            handle.write(json.dumps(task, ensure_ascii=True) + "\n")
    print(f"Wrote {len(prepared)} CUA web task(s) to {output}")
    for task in prepared:
        print(f"  {task['id']} {task['app_type']} {task['difficulty']} {task['base_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
