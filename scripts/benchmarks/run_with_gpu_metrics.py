#!/usr/bin/env python3
"""Run a command while sampling GPU utilization with nvidia-smi.

The wrapper is intentionally generic so the expensive benchmark can stay close
to the real launch command:

    python scripts/benchmarks/run_with_gpu_metrics.py \
      --tag before --name swegym_smoke --rollout-gpus 0,1,2,3 --train-gpus 4,5,6,7 \
      -- bash examples/swegym_slime_grpo/run.sh

It writes raw samples to CSV and a compact JSON summary under
``tmp/benchmarks/gpu`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="local", help="label written into output filenames")
    parser.add_argument("--name", default="command", help="benchmark command name")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "tmp" / "benchmarks" / "gpu"),
        help="directory for CSV/JSON outputs",
    )
    parser.add_argument("--interval-s", type=float, default=1.0, help="sampling interval")
    parser.add_argument("--rollout-gpus", default="", help="comma-separated GPU ids used for rollout")
    parser.add_argument("--train-gpus", default="", help="comma-separated GPU ids used for training")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}_{args.tag}_{args.name}"
    csv_path = out_dir / f"{stem}_gpu.csv"
    json_path = out_dir / f"{stem}_summary.json"
    stdout_path = out_dir / f"{stem}_stdout.log"
    stderr_path = out_dir / f"{stem}_stderr.log"

    stop_event = threading.Event()
    samples: list[dict[str, Any]] = []
    sampler_error: list[str] = []
    sampler_thread = threading.Thread(
        target=_sample_gpu_loop,
        args=(stop_event, args.interval_s, samples, sampler_error),
        name="gpu-metrics-sampler",
        daemon=True,
    )

    start = time.monotonic()
    sampler_thread.start()
    with stdout_path.open("w") as stdout_fh, stderr_path.open("w") as stderr_fh:
        process = subprocess.Popen(
            args.command,
            cwd=ROOT,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )
        returncode = process.wait()
    elapsed = time.monotonic() - start
    stop_event.set()
    sampler_thread.join(timeout=max(2.0, args.interval_s * 2))

    _write_csv(csv_path, samples)
    summary = {
        "benchmark": "gpu_command",
        "command": args.command,
        "created_at": timestamp,
        "elapsed_s": elapsed,
        "name": args.name,
        "returncode": returncode,
        "stderr_path": str(stderr_path),
        "stdout_path": str(stdout_path),
        "tag": args.tag,
        "gpu_csv_path": str(csv_path),
        "sampler_error": sampler_error[-1] if sampler_error else None,
        "gpu_summary": _summarize_gpu(
            samples,
            rollout_gpus=_parse_gpu_ids(args.rollout_gpus),
            train_gpus=_parse_gpu_ids(args.train_gpus),
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    if sampler_error:
        print(f"gpu sampler warning: {sampler_error[-1]}", file=sys.stderr)
    return returncode


def _sample_gpu_loop(
    stop_event: threading.Event,
    interval_s: float,
    samples: list[dict[str, Any]],
    sampler_error: list[str],
) -> None:
    if shutil.which("nvidia-smi") is None:
        sampler_error.append("nvidia-smi not found")
        return
    query = (
        "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10.0)
            if proc.returncode != 0:
                sampler_error.append(proc.stderr.strip() or f"nvidia-smi exited {proc.returncode}")
            else:
                now = time.time()
                for line in proc.stdout.splitlines():
                    parsed = _parse_nvidia_smi_line(line)
                    if parsed is not None:
                        parsed["sample_time"] = now
                        samples.append(parsed)
        except Exception as exc:
            sampler_error.append(str(exc))
        stop_event.wait(interval_s)


def _parse_nvidia_smi_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 7:
        return None
    try:
        return {
            "timestamp": parts[0],
            "gpu": int(parts[1]),
            "util_gpu_pct": float(parts[2]),
            "util_mem_pct": float(parts[3]),
            "memory_used_mb": float(parts[4]),
            "memory_total_mb": float(parts[5]),
            "power_draw_w": float(parts[6]),
        }
    except ValueError:
        return None


def _write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_time",
        "timestamp",
        "gpu",
        "util_gpu_pct",
        "util_mem_pct",
        "memory_used_mb",
        "memory_total_mb",
        "power_draw_w",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key) for key in fieldnames})


def _summarize_gpu(
    samples: list[dict[str, Any]],
    *,
    rollout_gpus: set[int],
    train_gpus: set[int],
) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_gpu.setdefault(int(sample["gpu"]), []).append(sample)

    per_gpu = {
        str(gpu): _summarize_samples(gpu_samples)
        for gpu, gpu_samples in sorted(by_gpu.items())
    }
    return {
        "sample_count": len(samples),
        "per_gpu": per_gpu,
        "rollout": _summarize_group(samples, rollout_gpus),
        "train": _summarize_group(samples, train_gpus),
        "all": _summarize_samples(samples),
    }


def _summarize_group(samples: list[dict[str, Any]], gpu_ids: set[int]) -> dict[str, float] | None:
    if not gpu_ids:
        return None
    group_samples = [sample for sample in samples if int(sample["gpu"]) in gpu_ids]
    return _summarize_samples(group_samples)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "avg_gpu_util_pct": 0.0,
            "avg_memory_used_mb": 0.0,
            "avg_power_draw_w": 0.0,
            "p95_gpu_util_pct": 0.0,
        }
    util = sorted(float(sample["util_gpu_pct"]) for sample in samples)
    p95_idx = min(len(util) - 1, int(len(util) * 0.95))
    return {
        "avg_gpu_util_pct": sum(util) / len(util),
        "avg_memory_used_mb": sum(float(sample["memory_used_mb"]) for sample in samples) / len(samples),
        "avg_power_draw_w": sum(float(sample["power_draw_w"]) for sample in samples) / len(samples),
        "p95_gpu_util_pct": util[p95_idx],
    }


def _parse_gpu_ids(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part) for part in value.split(",") if part.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
