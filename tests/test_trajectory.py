"""Smoke tests for the trajectory module cleanups."""

from __future__ import annotations

import asyncio

import pytest

from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.evaluator._patch_utils import BasePatchEvaluator
from polar.trajectory.evaluator.swebench_harness import SwebenchHarnessEvaluator
from polar.trajectory.evaluator.test_on_output import (
    TestOnOutputEvaluator,
    _normalize_expected_nodeid,
    _parse_expected_output,
)
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace
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


# ---------------------------------------------------------------------------
# PrefixMergingBuilder — raw response + canonical interstitial
# ---------------------------------------------------------------------------


# Synthetic Qwen-style token ids.
_EOT = 9000           # <|im_end|>
_IM_START = 9001      # <|im_start|>
_NL = 10              # \n
_SYS_PROMPT = [1, 2, 3]                       # canonical <|im_start|>system...<|im_end|>\n
_USER_PROMPT = [4, 5, 6, 7]                   # canonical <|im_start|>user...<|im_end|>\n
_GEN_PROMPT = [_IM_START, 100, _NL]           # <|im_start|>assistant\n


def _canonical_prompt_ids(asst_turns: list[list[int]], tools: list[list[int]]) -> list[int]:
    """Render canonical prompt ids for a chain up to the N-th generation prompt.

    asst_turns[i] = canonical body of the i-th assistant message (no EOT).
    tools[i]      = canonical tokens for the i-th tool-response message
                    (full <|im_start|>tool...<|im_end|>\\n block).
    Returns: sys + user + sum_i(asst_i + EOT + \\n + tools[i]) + gen_prompt
    """
    out = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    for asst, tool in zip(asst_turns, tools):
        out.extend(asst)
        out.append(_EOT)
        out.append(_NL)
        out.extend(tool)
        out.extend(_GEN_PROMPT)
    return out


def _make_record(
    cid: str,
    prompt_ids: list[int],
    prompt_messages: list[dict],
    response_ids: list[int],
    response_message: dict,
    finish_reason: str = "stop",
) -> CompletionRecord:
    logprobs_content = [
        {"token_id": tid, "token": f"<t{tid}>", "logprob": -0.1}
        for tid in response_ids
    ]
    return CompletionRecord(
        completion_id=cid,
        timestamp=cid,
        request={"messages": prompt_messages},
        response={
            "choices": [
                {
                    "input_token_ids": prompt_ids,
                    "token_ids": response_ids,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    "logprobs": {"content": logprobs_content},
                }
            ]
        },
    )


def _run_builder(builder: PrefixMergingBuilder, completions: list[CompletionRecord]):
    session = CompletionSession(session_id="s1", completions=completions)
    return asyncio.run(builder.build(session))


def test_prefix_merging_merges_raw_response_and_canonical_interstitial() -> None:
    """Happy path: two-turn chain stitches raw_1 + canonical tool interstitial + raw_2."""
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    asst1_raw = [200, 201, _EOT]
    asst1_msg = {"role": "assistant", "content": "a1"}
    tool_block = [_IM_START, 300, _NL, 301, _EOT, _NL]
    tool_msg = {"role": "tool", "content": "t1"}
    asst2_raw = [400, 401, _EOT]
    asst2_msg = {"role": "assistant", "content": "a2"}

    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = _canonical_prompt_ids([[200, 201]], [tool_block])

    c1 = _make_record("c1", c1_prompt, sys_user, asst1_raw, asst1_msg)
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [asst1_msg, tool_msg],
        asst2_raw,
        asst2_msg,
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    stats = traj.metadata["reconstruction_stats"]
    assert stats["chains_total"] == 1
    assert stats["completions_merged"] == 2
    assert stats["chains_reconstructed_full"] == 1

    trace = traj.traces[0]
    assert trace.prompt_ids == c1_prompt
    # response = raw_1 + canonical interstitial (skipping duplicate EOT)
    # + raw_2.  Because raw ends with _EOT, interstitial drops tail[0].
    expected_interstitial = [_NL] + list(tool_block) + list(_GEN_PROMPT)
    assert trace.response_ids == asst1_raw + expected_interstitial + asst2_raw


def test_prefix_merging_survives_bpe_drift_inside_assistant_body() -> None:
    """BPE drift: canonical and raw diverge at position 1 but bytes agree.

    canonical(asst1) = [200, 201]     (BPE-merged)
    raw(asst1)       = [200, 299, _EOT]   — model emitted a non-canonical merge path
    The raw path must survive stitching since only it has real logprobs.
    """
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    asst1_canonical = [200, 201]
    asst1_raw = [200, 299, _EOT]           # non-canonical but byte-equal
    asst1_msg = {"role": "assistant", "content": "a1"}
    tool_block = [_IM_START, 300, _EOT, _NL]
    tool_msg = {"role": "tool", "content": "t1"}
    asst2_raw = [400, _EOT]
    asst2_msg = {"role": "assistant", "content": "a2"}

    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = _canonical_prompt_ids([asst1_canonical], [tool_block])

    c1 = _make_record("c1", c1_prompt, sys_user, asst1_raw, asst1_msg)
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [asst1_msg, tool_msg],
        asst2_raw,
        asst2_msg,
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    stats = traj.metadata["reconstruction_stats"]
    assert stats["completions_merged"] == 2
    assert stats["chains_reconstructed_full"] == 1

    trace = traj.traces[0]
    # The raw (non-canonical) assistant body is preserved as-is; canonical
    # interstitial is tacked on after.
    expected_interstitial = [_NL] + list(tool_block) + list(_GEN_PROMPT)
    assert trace.response_ids == asst1_raw + expected_interstitial + asst2_raw


def test_prefix_merging_splits_on_token_prefix_divergence() -> None:
    """Message-key matches but raw tokens diverge (harness rewrote earlier
    content, e.g. cache_control shift / system-reminder injection).  Chain
    detection must refuse to join these into one chain — they should become
    two independent chains instead of a single truncated one.
    """
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    # c2's message_key would match c1 + c1's response (tool role is stripped
    # by _grouping_key) but its raw prompt tokens start differently.
    c2_prompt = (
        [99, 99, 99]
        + list(_USER_PROMPT)
        + list(_GEN_PROMPT)
        + [200, _EOT, _NL, _IM_START, 500, _EOT, _NL]
        + list(_GEN_PROMPT)
    )

    c1 = _make_record("c1", c1_prompt, sys_user, [200, _EOT], {"role": "assistant", "content": "a1"})
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t"}],
        [400, _EOT],
        {"role": "assistant", "content": "a2"},
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    stats = traj.metadata["reconstruction_stats"]
    # Two independent chains, each single-completion and trivially "full".
    assert stats["chains_total"] == 2
    assert stats["chains_reconstructed_full"] == 2
    assert stats["chains_reconstructed_truncated"] == 0
    assert stats["completions_merged"] == 2
    assert len(traj.traces) == 2
    # First trace = C_1 alone.  Second trace = C_2 alone (new chain root).
    assert traj.traces[0].response_ids == [200, _EOT]
    assert traj.traces[1].response_ids == [400, _EOT]


def test_prefix_merging_swegym_pattern_splits_instead_of_collapsing() -> None:
    """Regression test for the SWE-gym pattern: 3 completions whose
    message_keys all match append-only semantics, but whose raw prompt_ids
    diverge after C_1 (simulating cache-control / tools-schema shifts).
    Previously this collapsed into one chain whose finalize truncated to
    the first turn — misattributing the session reward to the opening
    assistant message.  The fix splits them into three independent chains.
    """
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]

    # All three prompts look message-equivalent after _grouping_key strips
    # tool_response entries, but each has a slightly different opening token
    # (simulating harness perturbation between turns).
    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = [9001, 9002] + c1_prompt[2:] + [200, _EOT, _NL] + list(_GEN_PROMPT)
    c3_prompt = [9003, 9004] + c2_prompt[2:] + [400, _EOT, _NL] + list(_GEN_PROMPT)

    c1 = _make_record("c1", c1_prompt, sys_user, [200, _EOT], {"role": "assistant", "content": "a1"})
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t1"}],
        [400, _EOT],
        {"role": "assistant", "content": "a2"},
    )
    c3 = _make_record(
        "c3",
        c3_prompt,
        sys_user
        + [
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "content": "t2"},
        ],
        [500, _EOT],
        {"role": "assistant", "content": "a3"},
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2, c3])
    stats = traj.metadata["reconstruction_stats"]
    assert stats["chains_total"] == 3
    assert stats["chains_reconstructed_full"] == 3
    assert stats["chains_reconstructed_truncated"] == 0
    assert stats["completions_merged"] == 3


def test_prefix_merging_handles_truncated_response_without_eot() -> None:
    """finish_reason=length → raw has no EOT → interstitial must prepend it."""
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    asst1_raw = [200, 201]                 # no _EOT (truncated)
    asst1_msg = {"role": "assistant", "content": "a1"}
    tool_block = [_IM_START, 300, _EOT, _NL]
    tool_msg = {"role": "tool", "content": "t"}
    asst2_raw = [400, _EOT]
    asst2_msg = {"role": "assistant", "content": "a2"}

    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = _canonical_prompt_ids([[200, 201]], [tool_block])

    c1 = _make_record(
        "c1", c1_prompt, sys_user, asst1_raw, asst1_msg, finish_reason="length"
    )
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [asst1_msg, tool_msg],
        asst2_raw,
        asst2_msg,
    )

    # Explicit EOT config (auto-detect would skip the length-finish c1 and pick
    # c2, which also works, but we pin it here for clarity).
    builder = PrefixMergingBuilder(end_of_turn_token_id=_EOT)
    traj = _run_builder(builder, [c1, c2])
    trace = traj.traces[0]
    # Interstitial should START with _EOT (since raw was truncated).
    expected_interstitial = [_EOT, _NL] + list(tool_block) + list(_GEN_PROMPT)
    assert trace.response_ids == asst1_raw + expected_interstitial + asst2_raw
    assert traj.metadata["reconstruction_stats"]["completions_merged"] == 2


def test_prefix_merging_auto_detects_eot_from_natural_stop() -> None:
    """Without end_of_turn_token_id config, builder should sniff it from C_1."""
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    asst1_raw = [200, 201, _EOT]
    tool_block = [_IM_START, 300, _EOT, _NL]
    asst2_raw = [400, _EOT]

    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = _canonical_prompt_ids([[200, 201]], [tool_block])

    c1 = _make_record("c1", c1_prompt, sys_user, asst1_raw, {"role": "assistant", "content": "a1"})
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t"}],
        asst2_raw,
        {"role": "assistant", "content": "a2"},
    )

    # No explicit eot id — builder must auto-detect it as _EOT from c1's last token.
    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    assert traj.metadata["reconstruction_stats"]["completions_merged"] == 2


def test_prefix_merging_merges_even_when_first_prompt_has_preamble() -> None:
    """§2 fix: harness-seeded preamble (system, user, user, assistant, tool)
    used to fall back to last-trace-only; v3 should now keep the chain and
    split at len(C_1.prompt_ids).
    """
    sys_user_preamble = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "preamble-asst"},
        {"role": "tool", "content": "preamble-tool"},
    ]
    # Canonical prompt for C_1 — pretend the server tokenized the preamble.
    c1_prompt_ids = list(_SYS_PROMPT) + list(_USER_PROMPT) + [50, 51, 52, 53] + list(_GEN_PROMPT)
    asst1_raw = [200, _EOT]
    asst1_msg = {"role": "assistant", "content": "a1"}
    tool_block = [_IM_START, 300, _EOT, _NL]
    tool_msg = {"role": "tool", "content": "t1"}
    asst2_raw = [400, _EOT]
    asst2_msg = {"role": "assistant", "content": "a2"}

    c2_prompt_ids = list(c1_prompt_ids) + [200, _EOT, _NL] + list(tool_block) + list(_GEN_PROMPT)

    c1 = _make_record("c1", c1_prompt_ids, sys_user_preamble, asst1_raw, asst1_msg)
    c2 = _make_record(
        "c2",
        c2_prompt_ids,
        sys_user_preamble + [asst1_msg, tool_msg],
        asst2_raw,
        asst2_msg,
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    stats = traj.metadata["reconstruction_stats"]
    assert stats["chains_reconstructed_full"] == 1
    assert stats["completions_merged"] == 2

    trace = traj.traces[0]
    # Prompt split = C_1.prompt_ids as-is (preamble included).
    assert trace.prompt_ids == c1_prompt_ids
    # Response = raw_1 + interstitial + raw_2.  Raw_1 ends in _EOT so
    # interstitial skips tail[0].
    expected_interstitial = [_NL] + list(tool_block) + list(_GEN_PROMPT)
    assert trace.response_ids == asst1_raw + expected_interstitial + asst2_raw


def test_prefix_merging_logprobs_layout_real_then_interstitial_then_real() -> None:
    """Interstitial positions get logprob=0.0 (masked); raw positions keep real."""
    sys_user = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    asst1_raw = [200, 201, _EOT]
    tool_block = [_IM_START, 300, _EOT, _NL]
    asst2_raw = [400, _EOT]

    c1_prompt = list(_SYS_PROMPT) + list(_USER_PROMPT) + list(_GEN_PROMPT)
    c2_prompt = _canonical_prompt_ids([[200, 201]], [tool_block])

    c1 = _make_record("c1", c1_prompt, sys_user, asst1_raw, {"role": "assistant", "content": "a1"})
    c2 = _make_record(
        "c2",
        c2_prompt,
        sys_user + [{"role": "assistant", "content": "a1"}, {"role": "tool", "content": "t"}],
        asst2_raw,
        {"role": "assistant", "content": "a2"},
    )

    traj = _run_builder(PrefixMergingBuilder(), [c1, c2])
    trace = traj.traces[0]
    assert trace.response_logprobs is not None
    assert len(trace.response_logprobs) == len(trace.response_ids)
    # raw_1 slots have real (-0.1) logprobs.
    for pos in range(len(asst1_raw)):
        assert trace.response_logprobs[pos]["logprob"] == -0.1
    # raw_2 slots also real.
    raw2_start = len(trace.response_ids) - len(asst2_raw)
    for pos in range(raw2_start, len(trace.response_ids)):
        assert trace.response_logprobs[pos]["logprob"] == -0.1
    # Interstitial slots between them have zero logprob.
    for pos in range(len(asst1_raw), raw2_start):
        assert trace.response_logprobs[pos]["logprob"] == 0.0
