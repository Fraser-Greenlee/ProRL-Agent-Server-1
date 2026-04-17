"""Smoke tests for the trajectory module cleanups."""

from __future__ import annotations

import pytest

from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.evaluator._patch_utils import BasePatchEvaluator
from polar.trajectory.evaluator.swebench_harness import SwebenchHarnessEvaluator
from polar.trajectory.evaluator.test_on_output import (
    TestOnOutputEvaluator,
    _normalize_expected_nodeid,
    _parse_expected_output,
)
from polar.trajectory.models import CompletionRecord, Trace
from polar.trajectory.registry import default_evaluator_registry


# ---------------------------------------------------------------------------
# Trace.tools removal
# ---------------------------------------------------------------------------


def test_trace_has_no_tools_field() -> None:
    assert "tools" not in Trace.model_fields


def test_build_trace_from_completion_ignores_tools_in_request() -> None:
    record = CompletionRecord(
        completion_id="c1",
        request={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
        },
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    trace = build_trace_from_completion(record)
    # The tools entry in the request must not leak onto the trace.
    assert not hasattr(trace, "tools")
    assert trace.prompt_messages == [{"role": "user", "content": "hi"}]
    assert trace.response_messages == [{"role": "assistant", "content": "hello"}]


# ---------------------------------------------------------------------------
# Registry exposes the split strategies under their new names
# ---------------------------------------------------------------------------


def test_default_registry_has_new_strategy_names() -> None:
    registry = default_evaluator_registry()
    assert set(registry.list_strategies()) == {
        "session_completed",
        "swebench_harness",
        "test_on_output",
    }


def test_default_registry_drops_old_combined_name() -> None:
    registry = default_evaluator_registry()
    assert "swegym_git_diff" not in registry.list_strategies()


# ---------------------------------------------------------------------------
# Config-level guardrails on the two strategies
# ---------------------------------------------------------------------------


def test_swebench_harness_requires_instance() -> None:
    with pytest.raises(ValueError, match="instance"):
        SwebenchHarnessEvaluator(instance={})


def test_test_on_output_requires_test_command() -> None:
    with pytest.raises(ValueError, match="test_command"):
        TestOnOutputEvaluator(test_command=" ", expected_output_json={"x": "PASSED"})


def test_test_on_output_requires_expected_output_json() -> None:
    with pytest.raises(ValueError, match="expected_output_json"):
        TestOnOutputEvaluator(test_command="pytest", expected_output_json=None)


# ---------------------------------------------------------------------------
# Shared BasePatchEvaluator helpers
# ---------------------------------------------------------------------------


def _make_test_on_output() -> TestOnOutputEvaluator:
    return TestOnOutputEvaluator(
        test_command="pytest",
        expected_output_json={"foo.test_a": "PASSED"},
    )


def test_base_patch_evaluator_filters_pycache_sections() -> None:
    evaluator = _make_test_on_output()
    patch = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@\n+x\n"
        "diff --git a/__pycache__/main.cpython-311.pyc b/__pycache__/main.cpython-311.pyc\n"
        "--- a/__pycache__/main.cpython-311.pyc\n"
        "+++ b/__pycache__/main.cpython-311.pyc\n"
        "@@\n+bytes\n"
    )
    filtered = evaluator._filter_patch(patch)
    assert "src/main.py" in filtered
    assert "__pycache__" not in filtered


def test_base_patch_evaluator_rejects_empty_repo_dir() -> None:
    with pytest.raises(ValueError, match="repo_dir"):
        TestOnOutputEvaluator(
            repo_dir=" ",
            test_command="pytest",
            expected_output_json={"x": "PASSED"},
        )


def test_base_patch_evaluator_rejects_non_positive_timeouts() -> None:
    with pytest.raises(ValueError, match="timeouts"):
        TestOnOutputEvaluator(
            test_command="pytest",
            expected_output_json={"x": "PASSED"},
            apply_timeout=0.0,
        )


def test_base_patch_evaluator_subclass_contract() -> None:
    # Both concrete strategies subclass the shared base so the skeleton
    # (extract → filter → apply → grade) is reused.
    assert issubclass(SwebenchHarnessEvaluator, BasePatchEvaluator)
    assert issubclass(TestOnOutputEvaluator, BasePatchEvaluator)


# ---------------------------------------------------------------------------
# test_on_output: node-id normalization + exact-match resolved rule
# ---------------------------------------------------------------------------


def test_normalize_expected_nodeid_trims_module_prefix() -> None:
    assert (
        _normalize_expected_nodeid("tests/test_calc.py::TestCalc::test_add")
        == "TestCalc.test_add"
    )
    assert _normalize_expected_nodeid("tests/test_calc.py::test_add") == "test_add"
    assert _normalize_expected_nodeid("test_add") == "test_add"


def test_parse_expected_output_picks_up_status_lines() -> None:
    output = (
        "collecting ...\n"
        "PASSED tests/test_calc.py::TestCalc::test_add\n"
        "FAILED tests/test_calc.py::TestCalc::test_sub - AssertionError\n"
        "some other line\n"
        "SKIPPED tests/test_calc.py::TestCalc::test_mul\n"
    )
    parsed = _parse_expected_output(output)
    assert parsed == {
        "TestCalc.test_add": "PASSED",
        "TestCalc.test_sub": "FAILED",
        "TestCalc.test_mul": "SKIPPED",
    }


def test_parse_expected_output_strips_ansi_escapes() -> None:
    output = "\x1b[32mPASSED\x1b[0m tests/test_calc.py::test_add\r\n"
    assert _parse_expected_output(output) == {"test_add": "PASSED"}


def test_test_on_output_coerces_json_string_to_dict() -> None:
    evaluator = TestOnOutputEvaluator(
        test_command="pytest",
        expected_output_json='{"tests/test_calc.py::TestCalc::test_add": "PASSED"}',
    )
    assert evaluator._coerce_expected_output_json() == {"TestCalc.test_add": "PASSED"}
