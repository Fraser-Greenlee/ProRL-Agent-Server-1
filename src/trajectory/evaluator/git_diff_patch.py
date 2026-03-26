"""Patch-based evaluator for SWE and r2e-gym style benchmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from runtime.base import BaseRuntime
from runtime.factory import create_runtime
from runtime.models import RuntimeSpec
from trajectory.evaluator.base import BaseTrajectoryEvaluator
from trajectory.models import EvalResult, Trajectory

_APPLY_PATCH_PASS = "__ARP_APPLY_PATCH_PASS__"
_APPLY_PATCH_FAIL = "__ARP_APPLY_PATCH_FAIL__"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\r")


class GitDiffPatchEvaluator(BaseTrajectoryEvaluator):
    """Evaluate a git diff patch against a prepared runtime."""

    def __init__(
        self,
        *,
        benchmark: str = "swe",
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        test_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        instance: dict[str, Any] | None = None,
        expected_output_json: dict[str, str] | str | None = None,
    ) -> None:
        self.benchmark = benchmark.strip().lower()
        if self.benchmark not in {"swe", "r2egym"}:
            raise ValueError("benchmark must be either 'swe' or 'r2egym'")
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
        self.test_command = (
            test_command.strip()
            if test_command is not None
            else "ln -s /r2e_tests /testbed/r2e_tests && bash /testbed/run_tests.sh"
        )
        if not self.test_command:
            raise ValueError("test_command must be non-empty")
        self.apply_timeout = float(apply_timeout)
        self.test_timeout = float(test_timeout)
        if self.apply_timeout <= 0 or self.test_timeout <= 0:
            raise ValueError("timeouts must be greater than 0")
        self.instance = instance or {}
        self.expected_output_json = expected_output_json

    async def evaluate(
        self,
        trajectory: Trajectory,
        **runtime: Any,
    ) -> EvalResult:
        source_runtime = runtime.get("runtime")
        if not isinstance(source_runtime, BaseRuntime):
            raise RuntimeError("git_diff_patch evaluator requires a live runtime")

        runtime_spec = runtime.get("runtime_spec")
        if runtime_spec is not None and not isinstance(runtime_spec, RuntimeSpec):
            raise TypeError("runtime_spec must be a RuntimeSpec when provided")

        session_id = str(runtime.get("session_id", "unknown-session"))
        task_id = str(runtime.get("task_id", "unknown-task"))
        session_dir = Path(runtime["session_dir"])
        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        patch_path = artifacts_dir / "git_diff_patch.diff"
        patch = await self._extract_patch(source_runtime, patch_path)
        metadata: dict[str, Any] = {
            "benchmark": self.benchmark,
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
        eval_runtime_root: Path | None = None
        eval_session_dir = session_dir
        eval_artifacts_dir = artifacts_dir
        apply_patch_output = ""
        created_fresh_runtime = False
        try:
            # Check if a fresh runtime was provided by the gateway
            fresh_runtime = runtime.get("fresh_eval_runtime")
            if isinstance(fresh_runtime, BaseRuntime):
                eval_runtime = fresh_runtime
                eval_runtime_root = artifacts_dir / "fresh_eval_runtime"
                eval_session_dir = eval_runtime_root / "session"
                eval_artifacts_dir = eval_session_dir / "artifacts"
                eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

                apply_patch_output = await self._apply_patch(
                    eval_runtime,
                    patch,
                    host_session_dir=eval_session_dir,
                    log_dir=eval_artifacts_dir,
                )
                metadata["apply_patch_output_path"] = str(eval_artifacts_dir / "apply_patch.stdout.log")
                if _APPLY_PATCH_FAIL in apply_patch_output or _APPLY_PATCH_PASS not in apply_patch_output:
                    metadata["report"]["failed_apply_patch"] = True
                    return EvalResult(outcome_reward=0.0, metadata=metadata)
            elif self.benchmark == "r2egym":
                metadata["report"]["failed_apply_patch"] = False

            if self.benchmark == "swe":
                report, test_output_path = await self._evaluate_swe(
                    eval_runtime,
                    patch,
                    host_session_dir=eval_session_dir,
                    log_dir=eval_artifacts_dir,
                )
            else:
                report, test_output_path = await self._evaluate_r2egym(
                    eval_runtime,
                    log_dir=eval_artifacts_dir,
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
        finally:
            if created_fresh_runtime:
                try:
                    await eval_runtime.stop()
                except Exception:
                    pass

    async def _extract_patch(self, runtime: BaseRuntime, patch_path: Path) -> str:
        result = await runtime.exec(self.patch_command, timeout_sec=self.apply_timeout)
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
        return result.stdout or ""

    async def _apply_patch(
        self,
        runtime: BaseRuntime,
        patch: str,
        *,
        host_session_dir: Path,
        log_dir: Path,
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
        result = await runtime.exec(apply_cmd, timeout_sec=self.apply_timeout)
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

    async def _evaluate_r2egym(
        self,
        runtime: BaseRuntime,
        *,
        log_dir: Path,
    ) -> tuple[dict[str, Any], Path]:
        combined_path = log_dir / "r2egym.test_output.log"
        result = await runtime.exec(self.test_command, timeout_sec=self.test_timeout)
        if result.return_code == -1:
            raise TimeoutError("r2egym evaluation timed out")
        output = (result.stdout or "") + (result.stderr or "")
        combined_path.write_text(output)
        expected = self._coerce_expected_output_json()
        parsed = self._parse_r2egym_output(output)
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

    async def _evaluate_swe(
        self,
        runtime: BaseRuntime,
        patch: str,
        *,
        host_session_dir: Path,
        log_dir: Path,
    ) -> tuple[dict[str, Any], Path]:
        if not self.instance:
            raise ValueError("swe benchmark evaluation requires an 'instance' config object")
        instance = dict(self.instance)
        instance_id = str(instance["instance_id"]).lower()
        instance["instance_id"] = instance_id
        if "version" not in instance and "base_commit" in instance:
            instance["version"] = instance["base_commit"]

        test_spec, get_eval_report = self._load_swe_harness(instance)
        eval_script_host = host_session_dir / "eval.sh"
        eval_script_host.write_text(test_spec.eval_script)

        combined_path = log_dir / "swe.test_output.log"
        result = await runtime.exec(
            f"/bin/bash {self._shell_quote(f'{runtime.runtime_session_dir}/eval.sh')}",
            timeout_sec=self.test_timeout,
        )
        if result.return_code == -1:
            raise TimeoutError("swe evaluation timed out")

        combined_path.write_text((result.stdout or "") + (result.stderr or ""))
        test_output_path = combined_path
        prediction = {"model_patch": patch, "instance_id": instance_id}
        grading_report = self._grade_swe_run(
            get_eval_report,
            test_spec=test_spec,
            prediction=prediction,
            log_path=test_output_path,
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
            test_output_path,
        )

    def _load_swe_harness(self, instance: dict[str, Any]) -> tuple[Any, Any]:
        try:
            from swegym.harness.grading import get_eval_report
            from swegym.harness.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.test_spec.test_spec import make_test_spec
        return make_test_spec(instance), get_eval_report

    @staticmethod
    def _grade_swe_run(get_eval_report: Any, *, test_spec: Any, prediction: dict[str, Any], log_path: Path) -> dict[str, Any]:
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
            raise ValueError("r2egym evaluation requires expected_output_json in evaluator config")
        raw = self.expected_output_json
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("expected_output_json must decode to a JSON object")
        return {
            self._normalize_r2egym_nodeid(str(key)): str(value)
            for key, value in parsed.items()
        }

    def _parse_r2egym_output(self, output: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in output.splitlines():
            line = _ANSI_ESCAPE_RE.sub("", line).strip()
            if not line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED")):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            status, nodeid = parts
            normalized = self._normalize_r2egym_nodeid(nodeid)
            if normalized:
                parsed[normalized] = status
        return parsed

    @staticmethod
    def _normalize_r2egym_nodeid(nodeid: str) -> str:
        nodeid = nodeid.split(" - ")[0]
        parts = nodeid.split("::")
        if len(parts) >= 3:
            return ".".join(parts[-2:])
        if len(parts) == 2:
            return parts[-1]
        return nodeid

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"
