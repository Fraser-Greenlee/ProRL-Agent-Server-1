"""Patch-based evaluator for fresh-runtime SWE-Gym style grading."""

from __future__ import annotations

import json
import fnmatch
import re
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.runtime.models import RuntimeSpec
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

_APPLY_PATCH_PASS = "__POLAR_APPLY_PATCH_PASS__"
_APPLY_PATCH_FAIL = "__POLAR_APPLY_PATCH_FAIL__"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\r")


class SweGymGitDiffEvaluator(BaseTrajectoryEvaluator):
    """Evaluate a git diff patch against the edited runtime or a fresh replay runtime."""

    def __init__(
        self,
        *,
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        test_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        instance: dict[str, Any] | None = None,
        expected_output_json: dict[str, str] | str | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.repo_dir = repo_dir.strip()
        if not self.repo_dir:
            raise ValueError("repo_dir must be non-empty")
        self.patch_command = (
            patch_command.strip()
            if patch_command is not None
            else f"cd {self._shell_quote(self.repo_dir)} && git diff --binary --submodule=diff"
        )
        if not self.patch_command:
            raise ValueError("patch_command must be non-empty")
        self.test_command = test_command.strip() if test_command is not None else ""
        self.apply_timeout = float(apply_timeout)
        self.test_timeout = float(test_timeout)
        if self.apply_timeout <= 0 or self.test_timeout <= 0:
            raise ValueError("timeouts must be greater than 0")
        self.instance = instance or {}
        self.expected_output_json = expected_output_json
        default_exclude_patterns = [
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
            "**/*.pyc",
            "*.pyo",
            "**/*.pyo",
            ".pytest_cache/**",
            "**/.pytest_cache/**",
        ]
        self.exclude_patterns = list(
            dict.fromkeys(
                [*default_exclude_patterns, *(exclude_patterns or [])]
            )
        )
        if not self.instance and self.expected_output_json is None:
            raise ValueError(
                "swegym_git_diff requires either 'instance' for SWE-Gym grading "
                "or 'expected_output_json' for expected-output grading"
            )

    async def evaluate(
        self,
        trajectory: Trajectory,
        **runtime: Any,
    ) -> EvalResult:
        source_runtime = runtime.get("runtime")
        if not isinstance(source_runtime, BaseRuntime):
            raise RuntimeError("swegym_git_diff evaluator requires a live runtime")

        runtime_spec = runtime.get("runtime_spec")
        if runtime_spec is not None and not isinstance(runtime_spec, RuntimeSpec):
            raise TypeError("runtime_spec must be a RuntimeSpec when provided")

        session_dir = Path(runtime["session_dir"])
        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        eval_env = runtime.get("env")
        if not isinstance(eval_env, dict):
            eval_env = {}
        timeout_cap = runtime.get("timeout_seconds")
        timeout_cap = float(timeout_cap) if timeout_cap is not None else None
        refresh_runtime = bool(runtime.get("refresh_runtime", False))

        patch_path = artifacts_dir / "swegym_git_diff.diff"
        patch = await self._extract_patch(
            source_runtime,
            patch_path,
            session_dir=session_dir,
            env=eval_env,
            timeout_cap=timeout_cap,
        )
        patch = self._filter_patch(patch)
        patch_path.write_text(patch)
        mode = "swegym" if self.instance else "expected_output"
        metadata: dict[str, Any] = {
            "mode": mode,
            "patch_path": str(patch_path),
            "report": {
                "empty_generation": len(patch.strip()) == 0,
                "resolved": False,
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
            },
        }
        if not patch.strip():
            return EvalResult(outcome_reward=0.0, metadata=metadata)

        eval_runtime: BaseRuntime = source_runtime
        eval_session_dir = session_dir
        eval_artifacts_dir = artifacts_dir
        apply_patch_output = ""
        try:
            fresh_runtime = runtime.get("fresh_eval_runtime")
            if isinstance(fresh_runtime, BaseRuntime):
                eval_runtime = fresh_runtime
                eval_session_dir = fresh_runtime.session_dir
                eval_artifacts_dir = fresh_runtime.artifacts_dir
                eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

                apply_patch_output = await self._apply_patch(
                    eval_runtime,
                    patch,
                    host_session_dir=eval_session_dir,
                    log_dir=eval_artifacts_dir,
                    env=eval_env,
                    timeout_cap=timeout_cap,
                )
                metadata["apply_patch_output_path"] = str(
                    eval_artifacts_dir / "apply_patch.stdout.log"
                )
                if (
                    _APPLY_PATCH_FAIL in apply_patch_output
                    or _APPLY_PATCH_PASS not in apply_patch_output
                ):
                    metadata["report"]["failed_apply_patch"] = True
                    return EvalResult(outcome_reward=0.0, metadata=metadata)
            elif refresh_runtime:
                raise RuntimeError(
                    "refresh_runtime=true requires fresh_eval_runtime for swegym_git_diff"
                )

            if self.instance:
                report, test_output_path = await self._evaluate_swegym(
                    eval_runtime,
                    patch,
                    host_session_dir=eval_session_dir,
                    log_dir=eval_artifacts_dir,
                    env=eval_env,
                    timeout_cap=timeout_cap,
                )
            else:
                report, test_output_path = await self._evaluate_expected_output(
                    eval_runtime,
                    log_dir=eval_artifacts_dir,
                    env=eval_env,
                    timeout_cap=timeout_cap,
                )
            metadata["report"] = report
            metadata["test_output_path"] = str(test_output_path)
            if apply_patch_output:
                metadata["apply_patch_output"] = apply_patch_output
            return EvalResult(
                outcome_reward=1.0 if report.get("resolved", False) else 0.0,
                metadata=metadata,
            )
        except TimeoutError:
            metadata["report"]["test_timeout"] = True
            raise
        except Exception:
            metadata["report"]["error_eval"] = True
            raise

    async def _extract_patch(
        self,
        runtime: BaseRuntime,
        patch_path: Path,
        *,
        session_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> str:
        result = await runtime.exec(
            self.patch_command,
            env=env,
            timeout_sec=self._bounded_timeout(self.apply_timeout, timeout_cap),
        )
        if result.stdout:
            patch_path.write_text(result.stdout)
        if result.stderr:
            patch_path.with_suffix(".stderr.log").write_text(result.stderr)
        if result.return_code == -1:
            raise TimeoutError("timed out while collecting git diff patch")
        if result.return_code != 0:
            raise RuntimeError(
                f"git diff command failed with exit code {result.return_code}: {result.stderr}"
            )
        patch = result.stdout or ""
        if patch.strip():
            return patch

        fallback_patch = self._read_fallback_patch(runtime, session_dir)
        if fallback_patch.strip():
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.with_suffix(".fallback.log").write_text(
                "git diff was empty; using saved agent patch artifact\n"
            )
            return fallback_patch
        return patch

    def _read_fallback_patch(self, runtime: BaseRuntime, session_dir: Path) -> str:
        for candidate in self._fallback_patch_candidates(runtime, session_dir):
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text()
            except OSError:
                continue
            if text.strip():
                return text
        return ""

    def _fallback_patch_candidates(
        self,
        runtime: BaseRuntime,
        session_dir: Path,
    ) -> list[Path]:
        candidates: list[Path] = [session_dir / "logs" / "agent" / "swe-agent.patch"]

        repo_host_dir = runtime.resolve_host_path(self.repo_dir)
        if repo_host_dir is not None:
            trajectories_dir = repo_host_dir / "trajectories"
            if trajectories_dir.exists():
                candidates.extend(
                    sorted(
                        trajectories_dir.rglob("*.patch"),
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                )
        return candidates

    async def _apply_patch(
        self,
        runtime: BaseRuntime,
        patch: str,
        *,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> str:
        patch_path = host_session_dir / "patch.diff"
        patch_path.write_text(patch)
        runtime_patch_path = f"{runtime.runtime_session_dir}/patch.diff"
        apply_cmd = (
            f"cd {self._shell_quote(self.repo_dir)} && "
            f"(git apply -v {self._shell_quote(runtime_patch_path)} && echo '{_APPLY_PATCH_PASS}' || "
            f"(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            f"(patch --batch --fuzz=5 -p1 -i {self._shell_quote(runtime_patch_path)} && "
            f"echo '{_APPLY_PATCH_PASS}' || echo '{_APPLY_PATCH_FAIL}')))"
        )
        result = await runtime.exec(
            apply_cmd,
            env=env,
            timeout_sec=self._bounded_timeout(self.apply_timeout, timeout_cap),
        )
        stdout_path = log_dir / "apply_patch.stdout.log"
        stderr_path = log_dir / "apply_patch.stderr.log"
        if result.stdout:
            stdout_path.write_text(result.stdout)
        if result.stderr:
            stderr_path.write_text(result.stderr)
        output = (result.stdout or "") + (result.stderr or "")
        if result.return_code == -1:
            raise TimeoutError("timed out while applying git diff patch")
        return output

    async def _evaluate_expected_output(
        self,
        runtime: BaseRuntime,
        *,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> tuple[dict[str, Any], Path]:
        if not self.test_command:
            raise ValueError(
                "expected-output grading requires 'test_command' in evaluator config"
            )
        combined_path = log_dir / "expected_output.test_output.log"
        result = await runtime.exec(
            self.test_command,
            env=env,
            timeout_sec=self._bounded_timeout(self.test_timeout, timeout_cap),
        )
        if result.return_code == -1:
            raise TimeoutError("expected-output evaluation timed out")
        output = (result.stdout or "") + (result.stderr or "")
        combined_path.write_text(output)
        expected = self._coerce_expected_output_json()
        parsed = self._parse_expected_output(output)
        report = {
            "empty_generation": False,
            "resolved": bool(parsed) and parsed == expected,
            "failed_apply_patch": False,
            "error_eval": False,
            "test_timeout": False,
            "exit_code": result.return_code,
            "parsed_tests": parsed,
            "expected_tests": expected,
        }
        return report, combined_path

    async def _evaluate_swegym(
        self,
        runtime: BaseRuntime,
        patch: str,
        *,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> tuple[dict[str, Any], Path]:
        if not self.instance:
            raise ValueError("swegym evaluation requires an 'instance' config object")
        instance = dict(self.instance)
        instance_id = str(instance["instance_id"]).lower()
        instance["instance_id"] = instance_id
        if "version" not in instance and "base_commit" in instance:
            instance["version"] = instance["base_commit"]

        test_spec, get_eval_report = self._load_swegym_harness(instance)
        eval_script_host = host_session_dir / "eval.sh"
        eval_script_host.write_text(test_spec.eval_script)

        # Place the log inside an instance_id-named directory so that
        # swegym/swebench get_logs_eval can parse the repo from the path.
        instance_log_dir = log_dir / instance_id
        instance_log_dir.mkdir(parents=True, exist_ok=True)
        combined_path = instance_log_dir / "test_output.txt"
        result = await runtime.exec(
            f"/bin/bash {self._shell_quote(f'{runtime.runtime_session_dir}/eval.sh')}",
            env=env,
            timeout_sec=self._bounded_timeout(self.test_timeout, timeout_cap),
        )
        if result.return_code == -1:
            raise TimeoutError("swegym evaluation timed out")

        combined_path.write_text((result.stdout or "") + (result.stderr or ""))
        prediction = {"model_patch": patch, "instance_id": instance_id}
        grading_report = self._grade_swegym_run(
            get_eval_report,
            test_spec=test_spec,
            prediction=prediction,
            log_path=combined_path,
        )
        report = grading_report[instance_id]
        return (
            {
                "empty_generation": False,
                "resolved": report.get("resolved", False),
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
                "exit_code": result.return_code,
                "grading_report": report,
            },
            combined_path,
        )

    def _load_swegym_harness(self, instance: dict[str, Any]) -> tuple[Any, Any]:
        try:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.test_spec.test_spec import make_test_spec
        return make_test_spec(instance), get_eval_report

    @staticmethod
    def _grade_swegym_run(
        get_eval_report: Any,
        *,
        test_spec: Any,
        prediction: dict[str, Any],
        log_path: Path,
    ) -> dict[str, Any]:
        try:
            return get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                log_path=str(log_path),
                include_tests_status=True,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                test_log_path=str(log_path),
                include_tests_status=True,
            )

    def _coerce_expected_output_json(self) -> dict[str, str]:
        if self.expected_output_json is None:
            raise ValueError(
                "expected-output grading requires expected_output_json in evaluator config"
            )
        raw = self.expected_output_json
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("expected_output_json must decode to a JSON object")
        return {
            self._normalize_expected_nodeid(str(key)): str(value)
            for key, value in parsed.items()
        }

    def _parse_expected_output(self, output: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in output.splitlines():
            line = _ANSI_ESCAPE_RE.sub("", line).strip()
            if not line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            status, nodeid = parts
            normalized = self._normalize_expected_nodeid(nodeid)
            if normalized:
                parsed[normalized] = status
        return parsed

    @staticmethod
    def _normalize_expected_nodeid(nodeid: str) -> str:
        nodeid = nodeid.split(" - ")[0]
        parts = nodeid.split("::")
        if len(parts) >= 3:
            return ".".join(parts[-2:])
        if len(parts) == 2:
            return parts[-1]
        return nodeid

    def _filter_patch(self, patch: str) -> str:
        if not patch.strip():
            return patch
        kept_sections: list[str] = []
        current_lines: list[str] = []
        for line in patch.splitlines(keepends=True):
            if line.startswith("diff --git "):
                if current_lines and not self._exclude_patch_section(current_lines):
                    kept_sections.extend(current_lines)
                current_lines = [line]
            elif current_lines:
                current_lines.append(line)
            else:
                kept_sections.append(line)
        if current_lines and not self._exclude_patch_section(current_lines):
            kept_sections.extend(current_lines)
        return "".join(kept_sections)

    def _exclude_patch_section(self, lines: list[str]) -> bool:
        for line in lines:
            if line.startswith("diff --git "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                    return self._matches_exclude(path)
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path != "/dev/null":
                    normalized = path[2:] if path.startswith("b/") else path
                    return self._matches_exclude(normalized)
        return False

    def _matches_exclude(self, path: str) -> bool:
        normalized = path.strip()
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(normalized, pattern):
                return True
        return False

    @staticmethod
    def _bounded_timeout(base_timeout: float, timeout_cap: float | None) -> float:
        if timeout_cap is None:
            return base_timeout
        return min(base_timeout, timeout_cap)

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"
