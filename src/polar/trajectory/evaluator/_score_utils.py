"""Shared scoring helpers for the answer-judge / env-state evaluators.

Both grade by parsing a numeric reward out of free-form command output (a JSON
object with a ``score``/``reward`` field, a ``score: 0.7`` line, or a bare
number) and clamping it to ``[0, 1]``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


def last_json_text(output: str) -> str:
    """Return the last line of *output* that looks like a JSON object/array."""
    text = output.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("{", "[")):
            return line
    return text


def parse_score(output: str) -> float:
    """Best-effort extraction of a numeric score from command output."""
    text = output.strip()
    if not text:
        return 0.0
    try:
        parsed = json.loads(last_json_text(text))
        if isinstance(parsed, dict):
            for key in ("score", "reward", "correct", "success"):
                if key in parsed:
                    value = parsed[key]
                    if isinstance(value, bool):
                        return 1.0 if value else 0.0
                    if isinstance(value, (int, float)):
                        return float(value)
        if isinstance(parsed, (int, float)):
            return float(parsed)
    except Exception:
        pass
    match = re.search(r"(?i)(?:score|reward)\s*[:=]\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    try:
        return float(text.splitlines()[-1].strip())
    except Exception:
        return 0.0


def lookup_path(value: Any, path: str) -> Any:
    """Resolve a dotted ``a.b.0.c`` path through nested dicts/lists."""
    current = value
    for raw_part in path.split("."):
        if raw_part == "":
            continue
        part: str | int = int(raw_part) if raw_part.isdigit() else raw_part
        if isinstance(part, int) and isinstance(current, list):
            current = current[part]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(path)
    return current


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))
