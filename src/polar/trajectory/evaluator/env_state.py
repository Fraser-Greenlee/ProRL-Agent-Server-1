"""``env_state`` evaluator — grade by inspecting the final environment state.

Loads the task's end state (HTTP GET a state URL, or run a state command in the
live runtime that prints JSON) and scores it via a numeric ``reward_key``, a
``success_path`` equality check, or a fallback ``reward_command`` that prints a
score. Used by the CUA-Gym web example, where state comes from the mock app's
``/go`` endpoint and ``reward.py`` grades it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator._patch_utils import bounded_timeout
from polar.trajectory.evaluator._score_utils import clamp01, last_json_text, lookup_path, parse_score
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory


class EnvStateEvaluator(BaseTrajectoryEvaluator):
    """Grade by inspecting the final environment state."""

    def __init__(
        self,
        *,
        state_url: str | None = None,
        state_command: str | None = None,
        sid: str | None = None,
        reward_key: str | None = None,
        reward_command: str | None = None,
        success_path: str | None = None,
        success_value: Any = True,
        request_timeout: float = 30.0,
        reward_timeout: float = 300.0,
    ) -> None:
        if not state_url and not state_command:
            raise ValueError("env-state requires state_url or state_command")
        self.state_url = state_url
        self.state_command = state_command
        self.sid = sid
        self.reward_key = reward_key
        self.reward_command = reward_command.strip() if reward_command else None
        self.success_path = success_path
        self.success_value = success_value
        self.request_timeout = float(request_timeout)
        self.reward_timeout = float(reward_timeout)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(runtime.get("session_id") or "")
        task_id = str(runtime.get("task_id") or "")
        sid = self.sid or session_id

        state = await self._load_state(
            runtime=runtime, sid=sid, session_id=session_id, task_id=task_id
        )
        state_path = artifacts_dir / "env_state.json"
        state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True))

        score, reason, reward_meta = self._score_from_state(state)
        if score is None and self.reward_command:
            score, reason, reward_meta = await self._score_with_command(
                state_path=state_path,
                runtime=runtime,
            )
        if score is None:
            score, reason = 0.0, "no reward_key/success_path/reward_command matched"

        return EvalResult(
            outcome_reward=clamp01(score),
            metadata={
                "mode": "env_state",
                "sid": sid,
                "state_path": str(state_path),
                "reason": reason,
                **reward_meta,
            },
        )

    async def _load_state(
        self,
        *,
        runtime: dict[str, Any],
        sid: str,
        session_id: str,
        task_id: str,
    ) -> Any:
        if self.state_url:
            url = self.state_url.format(sid=sid, session_id=session_id, task_id=task_id)
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()

        live_runtime = runtime.get("runtime")
        if not isinstance(live_runtime, BaseRuntime):
            raise RuntimeError("state_command requires a live runtime")
        command = (self.state_command or "").format(
            sid=sid,
            session_id=session_id,
            task_id=task_id,
        )
        result = await live_runtime.exec(
            command,
            timeout_sec=bounded_timeout(self.request_timeout, runtime.get("timeout_seconds")),
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.return_code == -1:
            raise TimeoutError("env-state state_command timed out")
        if result.return_code != 0:
            raise RuntimeError(
                f"state_command failed with exit code {result.return_code}: {output}"
            )
        return json.loads(last_json_text(output))

    def _score_from_state(self, state: Any) -> tuple[float | None, str, dict[str, Any]]:
        if self.reward_key:
            try:
                value = lookup_path(state, self.reward_key)
            except (KeyError, IndexError, TypeError) as exc:
                return 0.0, f"reward_key_missing={self.reward_key}", {"path_error": str(exc)}
            if isinstance(value, bool):
                return (1.0 if value else 0.0), f"reward_key={self.reward_key}", {"reward_value": value}
            if isinstance(value, (int, float)):
                return float(value), f"reward_key={self.reward_key}", {"reward_value": value}
        if self.success_path:
            try:
                value = lookup_path(state, self.success_path)
            except (KeyError, IndexError, TypeError) as exc:
                return 0.0, f"success_path_missing={self.success_path}", {"path_error": str(exc)}
            return (
                1.0 if value == self.success_value else 0.0,
                f"success_path={self.success_path}",
                {"success_value": value, "expected": self.success_value},
            )
        return None, "no_state_rule", {}

    async def _score_with_command(
        self,
        *,
        state_path: Path,
        runtime: dict[str, Any],
    ) -> tuple[float, str, dict[str, Any]]:
        live_runtime = runtime.get("runtime")
        if not isinstance(live_runtime, BaseRuntime):
            return 0.0, "reward_command requires live runtime", {}
        runtime_state_path = f"{live_runtime.runtime_artifacts_dir}/{state_path.name}"
        result = await live_runtime.exec(
            self.reward_command or "",
            env={"POLAR_STATE_PATH": runtime_state_path},
            timeout_sec=bounded_timeout(self.reward_timeout, runtime.get("timeout_seconds")),
        )
        output = (result.stdout or "") + (result.stderr or "")
        state_path.with_name("env_state.reward.log").write_text(output)
        if result.return_code == -1:
            raise TimeoutError("env-state reward_command timed out")
        if result.return_code != 0:
            return 0.0, f"reward_command exit_code={result.return_code}", {
                "reward_output": output[-4000:],
            }
        return parse_score(output), "reward_command", {"reward_output": output[-4000:]}
