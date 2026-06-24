"""Custom Polar evaluator for ARC-AGI compression tasks.

Subclasses :class:`BasePatchEvaluator`, so it inherits the hack-resistant flow
used by ``test_on_output`` / ``swebench_harness``: extract the agent's git diff
from the source runtime, drop everything except the one task file, apply it to a
*fresh* clean runtime, then grade there.  The agent's edits to anything other
than ``tasks/arcagi/train/<id>.hy`` never reach the grader.

Reward shape (overriding the base's 1.0/0.0 mapping):
- empty diff, failed apply, or not fully correct  → ``-1.0``
- fully correct (n_correct == n_total)            → ``(baseline_size - new_size)
  / baseline_size`` clipped to ``[0.0, 1.0]``

So every incorrect rollout (-1.0) ranks below every correct one (>= 0.0), and
correct rollouts form a continuum on compression.  A correct-but-not-smaller
solution scores 0.0.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator._patch_utils import (
    ANSI_ESCAPE_RE,
    BasePatchEvaluator,
    bounded_timeout,
)
from polar.trajectory.models import EvalResult, Trajectory

_CORRECT_RE = re.compile(r"^\s*\S+:\s*(\d+)\s*/\s*(\d+)\s*correct\s*$", re.M)
_INT_LINE_RE = re.compile(r"^\s*(\d+)\s*$")


class ArcCompressEvaluator(BasePatchEvaluator):
    """Grade an ARC-AGI compression rollout into a continuous reward."""

    MODE = "arc_compress"

    def __init__(
        self,
        *,
        task_id: str,
        baseline_size: int,
        task_kind: str = "arcagi",
        repo_dir: str = "/polar/session/workspace",
        synthetic_root: str = "/polar/session/workspace/sft/rl_tasks",
        patch_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 120.0,
        exclude_patterns: list[str] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            repo_dir=repo_dir,
            patch_command=patch_command,
            apply_timeout=apply_timeout,
            test_timeout=test_timeout,
            exclude_patterns=exclude_patterns,
        )
        if not task_id:
            raise ValueError("task_id is required")
        self.task_id = task_id
        self.baseline_size = int(baseline_size)
        if self.baseline_size <= 0:
            raise ValueError("baseline_size must be positive")
        if task_kind not in ("arcagi", "synthetic"):
            raise ValueError(f"task_kind must be arcagi|synthetic, got {task_kind!r}")
        self.task_kind = task_kind
        self.synthetic_root = synthetic_root
        # Correctness + size commands per task source. Both emit the same
        # formats the parsers expect: "<id>: n/N correct" and a bare integer.
        if task_kind == "arcagi":
            self._check_cmd = f"python eval.py --task arcagi/{task_id}"
            self._size_cmd = f"python eval.py --size arcagi/{task_id}"
        else:
            scorer = "sft/rl_tasks/score_synthetic.py"
            self._check_cmd = f"python {scorer} --root {synthetic_root} --check {task_id}"
            self._size_cmd = f"python {scorer} --root {synthetic_root} --size {task_id}"

    async def evaluate(
        self,
        trajectory: Trajectory,
        **runtime: Any,
    ) -> EvalResult:
        # Base class does extract → filter → apply-on-fresh-runtime → _grade,
        # and bakes outcome_reward = 1.0/0.0.  We rerun that mapping with our
        # -1 / [0,1] semantics, reading the numbers _grade stashed in the report.
        result = await super().evaluate(trajectory, **runtime)
        report = result.metadata.get("report", {})

        if report.get("empty_generation") or report.get("failed_apply_patch"):
            result.outcome_reward = -1.0
        elif report.get("error_eval"):
            result.outcome_reward = -1.0
        elif not report.get("resolved"):
            result.outcome_reward = -1.0
        else:
            result.outcome_reward = float(report.get("arc_reward", 0.0))
        return result

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
        log_path = log_dir / "arc_eval.log"
        report: dict[str, Any] = {
            "empty_generation": False,
            "resolved": False,
            "failed_apply_patch": False,
            "error_eval": False,
            "test_timeout": False,
            "task_id": self.task_id,
            "baseline_size": self.baseline_size,
            "n_correct": None,
            "n_total": None,
            "new_size": None,
            "arc_reward": -1.0,
        }

        # Correctness check → "<id>: nc/nt correct" (arcagi: eval.py --task;
        # synthetic: score_synthetic.py --check).
        correct_res = await runtime.exec(
            self._check_cmd,
            cwd=self.repo_dir,
            env=env,
            timeout_sec=bounded_timeout(self.test_timeout, timeout_cap),
        )
        if correct_res.return_code == -1:
            raise TimeoutError("arc_compress correctness check timed out")
        correct_out = (correct_res.stdout or "") + (correct_res.stderr or "")

        n_correct, n_total = _parse_correct(correct_out)
        report["n_correct"], report["n_total"] = n_correct, n_total

        size_out = ""
        if n_total is not None and n_correct == n_total:
            # Fully correct — measure the compressed object count.
            size_res = await runtime.exec(
                self._size_cmd,
                cwd=self.repo_dir,
                env=env,
                timeout_sec=bounded_timeout(self.test_timeout, timeout_cap),
            )
            if size_res.return_code == -1:
                raise TimeoutError("arc_compress size check timed out")
            size_out = (size_res.stdout or "") + (size_res.stderr or "")
            new_size = _parse_size(size_res.stdout or "")
            if new_size is not None:
                ratio = (self.baseline_size - new_size) / self.baseline_size
                report["resolved"] = True
                report["new_size"] = new_size
                report["arc_reward"] = max(0.0, min(1.0, ratio))
                report["reason"] = (
                    f"compressed {self.baseline_size} → {new_size}"
                    if new_size < self.baseline_size
                    else "correct but not smaller than baseline"
                )
            else:
                report["reason"] = "correct but could not parse --size output"
        else:
            report["reason"] = (
                f"failed correctness ({n_correct}/{n_total})"
                if n_total is not None
                else "could not parse correctness from eval.py"
            )

        log_path.write_text(
            f"$ {self._check_cmd}\n{correct_out}\n"
            f"$ {self._size_cmd}\n{size_out}\n"
        )
        return report, log_path


def _parse_correct(text: str) -> tuple[int | None, int | None]:
    m = _CORRECT_RE.search(ANSI_ESCAPE_RE.sub("", text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_size(text: str) -> int | None:
    # `eval.py --size` prints one integer on a line by itself.
    for line in ANSI_ESCAPE_RE.sub("", text).splitlines():
        m = _INT_LINE_RE.match(line)
        if m:
            return int(m.group(1))
    return None
