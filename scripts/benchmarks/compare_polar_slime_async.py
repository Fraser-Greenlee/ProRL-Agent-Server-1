#!/usr/bin/env python3
"""Compare two Polar/Slime async benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "elapsed_s",
    "groups_per_s",
    "sessions_per_s",
    "accepted_groups",
    "accepted_samples",
    "completed_tasks",
    "data_source_groups_produced",
    "max_server_inflight",
    "trainable_missing_logprob_samples",
    "positive_dummy_masks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", help="before aggregate/run JSON")
    parser.add_argument("after", help="after aggregate/run JSON")
    parser.add_argument("--out", help="optional markdown output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before = _load_runs(Path(args.before))
    after = _load_runs(Path(args.after))
    markdown = _render_markdown(before, after)
    if args.out:
        Path(args.out).write_text(markdown)
    print(markdown)
    return 0


def _load_runs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if "runs" in payload:
        return payload["runs"]
    if "scenarios" in payload:
        return [payload]
    raise ValueError(f"{path} is not a polar_slime_async benchmark JSON")


def _render_markdown(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    before_by_scenario = _group_by_scenario(before)
    after_by_scenario = _group_by_scenario(after)
    scenarios = sorted(set(before_by_scenario) | set(after_by_scenario))

    lines = [
        "# Polar/Slime Async Benchmark Compare",
        "",
        "| Scenario | Before Pass | After Pass | Before Time | After Time | Key Metric Delta |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scenario in scenarios:
        b = before_by_scenario.get(scenario, [])
        a = after_by_scenario.get(scenario, [])
        b_pass = _pass_rate(b)
        a_pass = _pass_rate(a)
        b_time = _mean(_scenario_values(b, "elapsed_s"))
        a_time = _mean(_scenario_values(a, "elapsed_s"))
        delta = _metric_delta(b, a)
        lines.append(
            f"| {scenario} | {b_pass:.0%} | {a_pass:.0%} | "
            f"{_fmt(b_time)} | {_fmt(a_time)} | {delta} |"
        )

    lines.extend([
        "",
        "Lower `elapsed_s`, fewer timeout/failure scenarios, higher throughput, and zero correctness counters are expected after the refactor.",
    ])
    return "\n".join(lines)


def _group_by_scenario(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for scenario in run.get("scenarios", []):
            grouped.setdefault(scenario["name"], []).append(scenario)
    return grouped


def _pass_rate(scenarios: list[dict[str, Any]]) -> float:
    if not scenarios:
        return 0.0
    return sum(1 for scenario in scenarios if scenario.get("passed")) / len(scenarios)


def _scenario_values(scenarios: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for scenario in scenarios:
        if key == "elapsed_s":
            value = scenario.get("elapsed_s")
        else:
            value = (scenario.get("metrics") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _metric_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    parts = []
    for key in METRIC_KEYS:
        if key == "elapsed_s":
            continue
        b_mean = _mean(_scenario_values(before, key))
        a_mean = _mean(_scenario_values(after, key))
        if b_mean is None and a_mean is None:
            continue
        parts.append(f"`{key}` {_fmt(b_mean)} -> {_fmt(a_mean)}")
    return "<br>".join(parts) if parts else "-"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
