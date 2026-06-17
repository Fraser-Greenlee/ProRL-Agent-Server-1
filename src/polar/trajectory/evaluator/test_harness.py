"""``test_harness`` evaluator — grade patch-producing coding tasks by a test cmd.

Generalises :mod:`polar.trajectory.evaluator.test_on_output`: run a configurable
test command in the patched repo and resolve via (a) an expected pytest-status
map, (b) exit code, and/or (c) success/failure regexes. Used by the R2E-Gym
example (run ``r2e_tests`` and check the gold outcomes).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator._patch_utils import (
    ANSI_ESCAPE_RE,
    BasePatchEvaluator,
    bounded_timeout,
)
from polar.trajectory.evaluator.test_on_output import _parse_expected_output


class TestHarnessEvaluator(BasePatchEvaluator):
    """Generic test-harness evaluator for patch-producing coding tasks."""

    MODE = "test_harness"

    def __init__(
        self,
        *,
        test_command: str,
        repo_dir: str = "/polar/session/workspace",
        patch_command: str | None = None,
        expected_output_json: dict[str, str] | str | None = None,
        success_regex: str | None = None,
        failure_regex: str | None = None,
        pass_exit_code: bool = True,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(
            repo_dir=repo_dir,
            patch_command=patch_command,
            apply_timeout=apply_timeout,
            test_timeout=test_timeout,
            exclude_patterns=exclude_patterns,
        )
        self.test_command = _required_text(test_command, "test_command")
        self.expected_output_json = expected_output_json
        self.success_regex = re.compile(success_regex, re.S) if success_regex else None
        self.failure_regex = re.compile(failure_regex, re.S) if failure_regex else None
        self.pass_exit_code = bool(pass_exit_code)

    async def _grade(
        self,
        *,
        runtime: BaseRuntime,
        patch: str,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> tuple[dict[str, Any], Path]:
        output_path = log_dir / "test_harness.output.log"
        result = await runtime.exec(
            self.test_command,
            cwd=self.repo_dir,
            env=env,
            timeout_sec=bounded_timeout(self.test_timeout, timeout_cap),
        )
        if result.return_code == -1:
            raise TimeoutError("test-harness evaluation timed out")

        output = (result.stdout or "") + (result.stderr or "")
        output_path.write_text(output)
        clean_output = ANSI_ESCAPE_RE.sub("", output)
        parsed_tests: dict[str, str] | None = None
        expected_tests: dict[str, str] | None = None

        if self.expected_output_json is not None:
            expected_tests = _coerce_json_object(self.expected_output_json)
            parsed_tests = _parse_expected_output(clean_output)
            resolved = bool(parsed_tests) and parsed_tests == expected_tests
        else:
            resolved = True
            if self.pass_exit_code:
                resolved = result.return_code == 0
            if self.success_regex is not None:
                resolved = resolved and bool(self.success_regex.search(clean_output))
            if self.failure_regex is not None:
                resolved = resolved and not bool(self.failure_regex.search(clean_output))

        return (
            {
                "empty_generation": False,
                "resolved": bool(resolved),
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
                "exit_code": result.return_code,
                "parsed_tests": parsed_tests,
                "expected_tests": expected_tests,
            },
            output_path,
        )


def _required_text(value: str, name: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    return cleaned


def _coerce_json_object(value: dict[str, str] | str) -> dict[str, str]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("expected_output_json must decode to an object")
    return {str(key): str(val) for key, val in parsed.items()}
