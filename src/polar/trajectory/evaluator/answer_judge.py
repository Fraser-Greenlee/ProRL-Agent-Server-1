"""``answer_judge`` evaluator — extract a final answer and match a reference.

Pulls the last ``<answer>...</answer>`` span out of the trajectory text and
scores it against one or more reference strings via exact / substring / numeric-
alias / regex matching, with an optional in-runtime judge command as a fallback.
Used by the LiteResearcher example.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator._patch_utils import bounded_timeout
from polar.trajectory.evaluator._score_utils import clamp01, parse_score
from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory


class AnswerJudgeEvaluator(BaseTrajectoryEvaluator):
    """Extract a final answer and compare it to a reference."""

    def __init__(
        self,
        *,
        reference: str | list[str],
        answer_regex: str = r"(?is)<answer>\s*(.*?)\s*</answer>",
        exact_match: bool = True,
        substring_match: bool = True,
        regex_match: str | None = None,
        judge_command: str | None = None,
        judge_timeout: float = 300.0,
    ) -> None:
        refs = reference if isinstance(reference, list) else [reference]
        self.references = [str(ref) for ref in refs]
        if not self.references:
            raise ValueError("answer-judge requires at least one reference")
        self.answer_regex = re.compile(answer_regex)
        self.exact_match = bool(exact_match)
        self.substring_match = bool(substring_match)
        self.regex_match = re.compile(regex_match, re.S) if regex_match else None
        self.judge_command = judge_command.strip() if judge_command else None
        self.judge_timeout = float(judge_timeout)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        artifacts_dir = Path(runtime["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        text = _collect_trajectory_text(trajectory)
        answer = self._extract_answer(text)
        (artifacts_dir / "answer_judge.prediction.txt").write_text(answer)
        (artifacts_dir / "answer_judge.references.json").write_text(
            json.dumps(self.references, ensure_ascii=True, indent=2)
        )

        score, reason, judge_meta = self._score_locally(answer)
        if score is None and self.judge_command:
            score, reason, judge_meta = await self._score_with_command(
                answer=answer,
                artifacts_dir=artifacts_dir,
                runtime=runtime,
            )
        if score is None:
            score, reason = 0.0, "no local match and no judge_command configured"

        return EvalResult(
            outcome_reward=clamp01(score),
            metadata={
                "mode": "answer_judge",
                "prediction": answer,
                "references": self.references,
                "reason": reason,
                **judge_meta,
            },
        )

    def _extract_answer(self, text: str) -> str:
        matches = list(self.answer_regex.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
        return text.strip().splitlines()[-1].strip() if text.strip() else ""

    def _score_locally(self, answer: str) -> tuple[float | None, str, dict[str, Any]]:
        normalized_answer = _normalize_text(answer)
        for ref in self.references:
            normalized_ref = _normalize_text(ref)
            if self.exact_match and normalized_answer == normalized_ref:
                return 1.0, "exact_match", {}
            if self.substring_match and normalized_ref and normalized_ref in normalized_answer:
                return 1.0, "substring_match", {}
            if _alias_match(answer, ref):
                return 1.0, "alias_match", {}
        if self.regex_match is not None and self.regex_match.search(answer):
            return 1.0, "regex_match", {}
        return None, "no_local_match", {}

    async def _score_with_command(
        self,
        *,
        answer: str,
        artifacts_dir: Path,
        runtime: dict[str, Any],
    ) -> tuple[float, str, dict[str, Any]]:
        live_runtime = runtime.get("runtime")
        if not isinstance(live_runtime, BaseRuntime):
            return 0.0, "judge_command requires live runtime", {}
        payload_path = artifacts_dir / "answer_judge.payload.json"
        payload_path.write_text(
            json.dumps(
                {"prediction": answer, "references": self.references},
                ensure_ascii=True,
            )
        )
        runtime_payload_path = f"{live_runtime.runtime_artifacts_dir}/{payload_path.name}"
        result = await live_runtime.exec(
            self.judge_command or "",
            env={"POLAR_JUDGE_PAYLOAD_PATH": runtime_payload_path},
            timeout_sec=bounded_timeout(self.judge_timeout, runtime.get("timeout_seconds")),
        )
        output = (result.stdout or "") + (result.stderr or "")
        (artifacts_dir / "answer_judge.command.log").write_text(output)
        if result.return_code == -1:
            raise TimeoutError("answer judge command timed out")
        if result.return_code != 0:
            return 0.0, f"judge_command exit_code={result.return_code}", {
                "judge_output": output[-4000:],
            }
        score = parse_score(output)
        return score, "judge_command", {"judge_output": output[-4000:]}


def _collect_trajectory_text(trajectory: Trajectory) -> str:
    chunks: list[str] = []
    for trace in trajectory.traces:
        for message in trace.response_messages:
            text = _message_text(message)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    return _content_text(message.get("content"))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ORDINAL_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}


def _alias_match(answer: str, reference: str) -> bool:
    answer_aliases = _answer_aliases(answer)
    reference_aliases = _answer_aliases(reference)
    if not answer_aliases or not reference_aliases:
        return False
    if answer_aliases & reference_aliases:
        return True
    for ref_alias in reference_aliases:
        if len(ref_alias) > 2 and any(ref_alias in answer_alias for answer_alias in answer_aliases):
            return True
    return False


def _answer_aliases(value: str) -> set[str]:
    simplified = re.sub(r"[^a-z0-9#]+", " ", value.casefold()).strip()
    simplified = re.sub(r"\s+", " ", simplified)
    if not simplified:
        return set()
    aliases = {simplified}
    tokens = simplified.split()

    replaced_tokens = []
    changed = False
    for token in tokens:
        replacement = _NUMBER_WORDS.get(token) or _ORDINAL_WORDS.get(token)
        if replacement is not None:
            replaced_tokens.append(replacement)
            changed = True
        else:
            replaced_tokens.append(token)
    if changed:
        aliases.add(" ".join(replaced_tokens))

    for word, number in _NUMBER_WORDS.items():
        if simplified in {word, f"number {word}", f"no {word}", f"# {word}"}:
            aliases.update({number, f"number {number}", f"no {number}", f"# {number}"})
    for word, number in _ORDINAL_WORDS.items():
        if simplified == word:
            aliases.add(number)
        if simplified == number:
            aliases.add(word)
    if simplified.startswith("#"):
        aliases.add(simplified[1:].strip())
    return aliases
