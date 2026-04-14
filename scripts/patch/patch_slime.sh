#!/usr/bin/env bash
# Patch Slime to accept pre-computed ("external") per-sample advantages
# from Polar's advantage estimators.
#
# Three files are touched:
#   1. slime/utils/arguments.py       – add "external" to --advantage-estimator choices
#   2. slime/ray/rollout.py           – extract external_advantages from sample metadata
#                                       and add to DP split whitelist
#   3. slime/backends/megatron_utils/loss.py – broadcast external scalars to per-token tensors
#
# Usage:
#   bash scripts/patch/patch_slime.sh

set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import importlib.util
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        fail(f"Failed to patch {label}: expected snippet not found")
    return text.replace(old, new, 1)


spec = importlib.util.find_spec("slime")
if spec is None or spec.origin is None:
    fail("slime is not installed in the active Python environment")

root = Path(spec.origin).resolve().parent

arguments_path = root / "utils/arguments.py"
rollout_path = root / "ray/rollout.py"
loss_path = root / "backends/megatron_utils/loss.py"

for path in (arguments_path, rollout_path, loss_path):
    if not path.exists():
        fail(f"Expected Slime file is missing: {path}")

# ── 1. arguments.py: add "external" to advantage-estimator choices ────────

arguments_text = arguments_path.read_text()
arguments_text = replace_once(
    arguments_text,
    '                choices=[\n'
    '                    "grpo",\n'
    '                    "gspo",\n'
    '                    "reinforce_plus_plus",\n'
    '                    "reinforce_plus_plus_baseline",\n'
    '                    "ppo",\n'
    '                ],\n',
    '                choices=[\n'
    '                    "grpo",\n'
    '                    "gspo",\n'
    '                    "reinforce_plus_plus",\n'
    '                    "reinforce_plus_plus_baseline",\n'
    '                    "ppo",\n'
    '                    "external",\n'
    '                ],\n',
    label=str(arguments_path),
)
arguments_path.write_text(arguments_text)

# ── 2. rollout.py: extract external_advantages + DP whitelist ─────────────

rollout_text = rollout_path.read_text()

# 2a. Extract per-sample advantages from metadata in _convert_samples_to_train_data
rollout_text = replace_once(
    rollout_text,
    '        if samples[0].teacher_log_probs is not None:\n'
    '            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]\n'
    '\n'
    '        return train_data\n',
    '        if samples[0].teacher_log_probs is not None:\n'
    '            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]\n'
    '\n'
    '        # Carry pre-computed advantages from external systems (e.g. Polar)\n'
    '        ext_advs = [\n'
    '            sample.metadata.get("polar", {}).get("advantage")\n'
    '            if sample.metadata else None\n'
    '            for sample in samples\n'
    '        ]\n'
    '        if any(a is not None for a in ext_advs):\n'
    '            train_data["external_advantages"] = [\n'
    '                float(a) if a is not None else 0.0 for a in ext_advs\n'
    '            ]\n'
    '\n'
    '        return train_data\n',
    label=f"{rollout_path} (_convert_samples_to_train_data)",
)

# 2b. Add "external_advantages" to the DP split whitelist
rollout_text = replace_once(
    rollout_text,
    '                "rollout_routed_experts",\n'
    '                "prompt",\n'
    '                "teacher_log_probs",\n'
    '            ]:\n',
    '                "rollout_routed_experts",\n'
    '                "prompt",\n'
    '                "teacher_log_probs",\n'
    '                "external_advantages",\n'
    '            ]:\n',
    label=f"{rollout_path} (_split_train_data_by_dp)",
)

rollout_path.write_text(rollout_text)

# ── 3. loss.py: add "external" advantage estimator branch ────────────────

loss_text = loss_path.read_text()
loss_text = replace_once(
    loss_text,
    '    else:\n'
    '        raise NotImplementedError('
    'f"advantage_estimator {args.advantage_estimator} is not supported. ")\n',

    '    elif args.advantage_estimator == "external":\n'
    '        # Pre-computed per-sample scalar advantages from an external system\n'
    '        # (e.g. Polar HierarchicalGroupAdvantageEstimator).  Broadcast each\n'
    '        # scalar to a per-token tensor, same layout as GRPO returns.\n'
    '        ext_advs = rollout_data.get("external_advantages")\n'
    '        if ext_advs is None:\n'
    '            raise ValueError(\n'
    '                "advantage_estimator=\'external\' requires \'external_advantages\' "\n'
    '                "in rollout_data (per-sample scalar list from the rollout side)."\n'
    '            )\n'
    '        device = kl[0].device if kl else torch.device("cuda")\n'
    '        ext_advs_t = torch.tensor(ext_advs, dtype=torch.float32, device=device)\n'
    '        returns = [torch.ones_like(kl[i]) * ext_advs_t[i] for i in range(len(ext_advs_t))]\n'
    '        advantages = [r for r in returns]\n'
    '\n'
    '    else:\n'
    '        raise NotImplementedError('
    'f"advantage_estimator {args.advantage_estimator} is not supported. ")\n',
    label=f"{loss_path} (compute_advantages_and_returns)",
)
loss_path.write_text(loss_text)

print(f"Patched Slime in {root}")
PY
