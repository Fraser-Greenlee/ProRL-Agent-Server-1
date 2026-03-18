#!/usr/bin/env python3
"""
Convert Kimi trajectories to LLaMA-Factory compatible SFT format.

Each action step becomes one training sample. The message structure mirrors
KimiActor._build_messages() exactly — system prompt, text-only old history,
recent history with screenshots (last 3), current screenshot + instruction.

The assistant response combines <think>reasoning</think> + raw model output.

Usage:
    python convert_to_sft.py \
        --input_dirs trajectories/kimi /path/to/more/trajectories/kimi \
        --output sft_data.jsonl \
        [--max_image_history_length 3] \
        [--password password] \
        [--thinking]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, "
    "task instruction and previous steps (if provided).\n"
)

STEP_TEMPLATE = "# Step {step_num}:\n"

# History entries are plain text (no thinking tokens — they are input context, not model output)
HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_input_screenshot(actions: List[Dict], action_idx: int, traj_dir: str) -> Optional[str]:
    """Get the screenshot the model saw BEFORE taking this action."""
    if action_idx == 0:
        initial = os.path.join(traj_dir, "0-0.png")
        return initial if os.path.isfile(initial) else None
    else:
        prev_screenshot = actions[action_idx - 1].get("screenshot", "")
        return prev_screenshot if prev_screenshot and os.path.isfile(prev_screenshot) else None


def build_assistant_response(action: Dict) -> str:
    """Build the assistant response: <think>reasoning</think> + action/code output."""
    raw_reasoning = action.get("raw_reasoning", "")
    raw_response = action.get("raw_response", "")

    if raw_reasoning:
        return f"<think>\n{raw_reasoning.strip()}\n</think>\n{raw_response.strip()}"
    else:
        return raw_response.strip()


def action_to_sample(
    goal: str,
    actions: List[Dict],
    action_idx: int,
    traj_dir: str,
    max_image_history_length: int,
) -> Optional[Dict]:
    """
    Convert a single action step into a LLaMA-Factory training sample.

    Mirrors KimiActor._build_messages() message construction:
    - Old history steps (beyond image window): text-only assistant message
    - Recent history steps (last max_image_history_length): user screenshot + assistant response
    - Current step: user screenshot + instruction -> assistant response with <think>...</think>
    """
    # --- Gather info ---
    current_action = actions[action_idx]

    # Current screenshot (what the model sees to decide this action)
    current_screenshot = get_input_screenshot(actions, action_idx, traj_dir)
    if current_screenshot is None:
        return None

    # Build history from all previous actions
    num_history = action_idx  # actions[0..action_idx-1]

    instruction_prompt = INSTRUCTION_TEMPLATE.format(instruction=goal)

    # --- Build messages and collect images ---
    images = []  # ordered list of image paths
    messages = []

    # History steps, following reference windowing logic
    old_history_texts = []
    for i in range(num_history):
        gen = actions[i].get("action_generation", {})
        thought = gen.get("thought", "")
        action_desc = gen.get("action", "")

        # Text-only history: plain thought/action (no thinking tokens — these are context)
        plain_step_text = STEP_TEMPLATE.format(step_num=i + 1) + HISTORY_TEMPLATE.format(
            thought=thought, action=action_desc
        )

        # Assistant response for recent history: includes <think>...</think> like actual model output
        history_response = build_assistant_response(actions[i])

        if i > num_history - max_image_history_length:
            # Recent step: include screenshot image + full assistant response with thinking
            hist_screenshot = get_input_screenshot(actions, i, traj_dir)
            if hist_screenshot is None:
                return None

            images.append(hist_screenshot)
            messages.append({
                "role": "user",
                "content": "<image>",
            })
            messages.append({
                "role": "assistant",
                "content": history_response,
            })
        else:
            # Old step: text-only, accumulate (no thinking tokens — just context summary)
            old_history_texts.append(plain_step_text)

            # Emit at boundary as user context (not assistant — this is input, not a generation target)
            if i == num_history - max_image_history_length:
                messages.append({
                    "role": "user",
                    "content": "# Previous steps:\n" + "\n".join(old_history_texts),
                })

    # --- Current step: user message with screenshot + instruction ---
    images.append(current_screenshot)
    messages.append({
        "role": "user",
        "content": f"<image>\n{instruction_prompt}",
    })

    # --- Assistant response (training target) ---
    assistant_response = build_assistant_response(current_action)
    if not assistant_response:
        return None

    # Quality filter: training target must have complete thinking block
    if "</think>" not in assistant_response:
        return None

    messages.append({
        "role": "assistant",
        "content": assistant_response,
    })

    # Validate all image paths exist
    for img_path in images:
        if not os.path.isfile(img_path):
            return None

    return {
        "images": [{"bytes": None, "path": p} for p in images],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def iter_trajectories(input_dirs: List[str]):
    """Iterate over all trajectory.json files in the given directories."""
    for base_dir in input_dirs:
        base = Path(base_dir)
        if not base.is_dir():
            print(f"WARNING: {base_dir} is not a directory, skipping.", file=sys.stderr)
            continue
        for collection_dir in sorted(base.iterdir()):
            if not collection_dir.is_dir():
                continue
            for traj_dir in sorted(collection_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                traj_json = traj_dir / "trajectory.json"
                if traj_json.is_file():
                    yield str(traj_dir), str(traj_json)


def main():
    parser = argparse.ArgumentParser(description="Convert Kimi trajectories to LLaMA-Factory SFT format")
    parser.add_argument(
        "--input_dirs", nargs="+", help="Directories containing trajectory collections",
        default=[
            "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/trajectories/kimi",
            "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/data/jaehun/cua/trajectories/kimi",
        ],
    )
    parser.add_argument(
        "--output", help="Output JSONL file path",
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/postprocess/20260317/sft.jsonl"
    )
    parser.add_argument("--max_image_history_length", type=int, default=3, help="Number of recent history screenshots to include")
    parser.add_argument("--skip_done_fail", action="store_true", help="Skip done/fail terminal actions")
    args = parser.parse_args()

    total_trajs = 0
    total_samples = 0
    skipped_trajs = 0
    skipped_actions = 0

    with open(args.output, "w") as fout:
        for traj_dir, traj_json_path in iter_trajectories(args.input_dirs):
            try:
                with open(traj_json_path) as f:
                    traj = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Failed to load {traj_json_path}: {e}", file=sys.stderr)
                skipped_trajs += 1
                continue

            goal = traj.get("goal", "")
            if not goal:
                skipped_trajs += 1
                continue

            actions = []
            for step in traj.get("steps", []):
                actions.extend(step.get("actions", []))

            if not actions:
                skipped_trajs += 1
                continue

            total_trajs += 1
            traj_samples = 0

            for action_idx, action in enumerate(actions):
                # Optionally skip terminal actions
                if args.skip_done_fail and action.get("action_type") in ("done", "fail"):
                    continue

                sample = action_to_sample(
                    goal=goal,
                    actions=actions,
                    action_idx=action_idx,
                    traj_dir=traj_dir,
                    max_image_history_length=args.max_image_history_length,
                )

                if sample is None:
                    skipped_actions += 1
                    continue

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                traj_samples += 1
                total_samples += 1

            if total_trajs % 1000 == 0:
                print(f"  Processed {total_trajs} trajectories, {total_samples} samples so far...", file=sys.stderr)

    print(f"\nDone.", file=sys.stderr)
    print(f"  Trajectories processed: {total_trajs}", file=sys.stderr)
    print(f"  Trajectories skipped:   {skipped_trajs}", file=sys.stderr)
    print(f"  Samples written:        {total_samples}", file=sys.stderr)
    print(f"  Actions skipped:        {skipped_actions}", file=sys.stderr)
    print(f"  Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
