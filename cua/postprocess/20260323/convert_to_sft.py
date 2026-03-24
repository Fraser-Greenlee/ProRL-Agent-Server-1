import json
from pathlib import Path
from typing import Dict, List

import ipdb
import jsonlines
from tqdm import tqdm


SYSTEM_PROMPT = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task.

For each step, provide your response in this format:
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()


INSTRUCTION_TEMPLATE = """
# Task Instruction:
{instruction}

Please generate the next move according to the screenshot, task instruction and previous steps (if provided).
""".strip()



def sample_to_sft_samples(sample: Dict) -> List[Dict]:
    """
    Turn each sample into series of ShareGPT SFT messages, flattening multi-turns.

    Output:
        List of sft samples, each formatted as:
        {"messages": [{"role": .., "content": ..}], "images": ["..jpg", ..]}
    """
    # flatten down sample
    flattened_samples = []
    for step_idx in range(len(sample["steps"])):
        # include up to step_idx
        sft_steps = sample["steps"][:step_idx + 1]
        sft_screenshots = sample["screenshots"][:step_idx + 1]
        if len(sft_screenshots) > 3:
            sft_screenshots = sft_screenshots[-3:]  # we only use the last 3 screenshots.

        flattened_samples.append({
            "instruction": sample["instruction"],
            "screenshots": sft_screenshots,
            "steps": sft_steps,
        })

    # format flattened sample into sft samples
    sft_samples = []
    for flattened_sample in flattened_samples:
        sft_samples.append(flattened_sample_to_sft_sample(flattened_sample))

    return sft_samples


def flattened_sample_to_sft_sample(flattened_sample: Dict) -> Dict:
    """
    Turn each flattened sample into a sft sample.

    LF parsing of ShareGPT format does not allow system prompt, so include it in the initial turn.
    Examples:
      Step 1 (first action, no history):
      [user]       SYSTEM_PROMPT + <screenshot 0-0> + instruction
      [assistant]  <think>...</think> response     ← target

      Step 2 (1 history step):
      [user]       SYSTEM_PROMPT + <screenshot 0-0>
      [assistant]  <think>...</think> Step 1 response
      [user]       <screenshot 0-1> + instruction
      [assistant]  <think>...</think> response     ← target

      Step 3 (2 history steps):
      [user]       SYSTEM_PROMPT + <screenshot 0-0>
      [assistant]  <think>...</think> Step 1 response
      [user]       <screenshot 0-1>
      [assistant]  <think>...</think> Step 2 response
      [user]       <screenshot 0-2> + instruction
      [assistant]  <think>...</think> response     ← target

      Step 4+ (old text history kicks in):
      [user]       SYSTEM_PROMPT + "Old steps:\nStep 1: thought+action" + <screenshot 0-1>
      [assistant]  <think>...</think> Step 2 response
      [user]       <screenshot 0-2>
      [assistant]  <think>...</think> Step 3 response
      [user]       <screenshot 0-3> + instruction
      [assistant]  <think>...</think> response     ← target

    Output:
        A dict, formatted as
        {"messages": [{"role": ..., "content": ...}], "images": ["...jpg", ...]}
    """
    num_steps = len(flattened_sample["steps"])  # includes the target step

    if num_steps == 1:  # No recent steps, no older steps
        messages = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n<image>\n{INSTRUCTION_TEMPLATE.format(instruction=flattened_sample['instruction'])}"
            },
            {
                "role": "assistant",
                "content": f"<think>{flattened_sample['steps'][-1]['reasoning'].strip()}</think>\n\n{flattened_sample['steps'][-1]['response'].strip()}"
            }
        ]
    elif num_steps < 4:  # Recent steps exist, but not older steps
        messages = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n<image>"
            }
        ]

        # add recent steps + last instruction
        for step_idx in range(0, num_steps - 1):
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{flattened_sample['steps'][step_idx]['reasoning'].strip()}</think>\n\n{flattened_sample['steps'][step_idx]['response'].strip()}"
                }
            )
            if step_idx < num_steps - 2:  # not the last user turn, so simply add <image>
                messages.append(
                    {
                        "role": "user",
                        "content": f"<image>"
                    }
                )
            else:  # last step, add instruction
                messages.append(
                    {
                        "role": "user",
                        "content": f"<image>\n{INSTRUCTION_TEMPLATE.format(instruction=flattened_sample['instruction'])}"
                    }
                )

        # target step
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{flattened_sample['steps'][-1]['reasoning'].strip()}</think>\n\n{flattened_sample['steps'][-1]['response'].strip()}"
            }
        )
    else:  # Recent steps exist, older steps exist
        # prepare old step (text only)
        old_step_texts = []
        for step_idx in range(0, num_steps - 3):
            reasoning = flattened_sample["steps"][step_idx]["reasoning"]
            response = flattened_sample["steps"][step_idx]["response"]
            old_step_texts.append(f"Step {step_idx + 1}:\nReasoning: {reasoning.strip()}\nResponse: {response.strip()}")

        old_step_str = '\n\n'.join(old_step_texts)
        old_step_str = f"Old steps:\n{old_step_str}"

        messages = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n{old_step_str}\n<image>"
            }
        ]

        # add recent steps + last instruction
        for step_idx in range(num_steps - 3, num_steps - 1):
            messages.append(
                {
                    "role": "assistant",
                    "content": f"<think>{flattened_sample['steps'][step_idx]['reasoning'].strip()}</think>\n\n{flattened_sample['steps'][step_idx]['response'].strip()}"
                }
            )
            if step_idx < num_steps - 2:  # not the last user turn, so simply add <image>
                messages.append(
                    {
                        "role": "user",
                        "content": f"<image>"
                    }
                )
            else:  # last step, add instruction
                messages.append(
                    {
                        "role": "user",
                        "content": f"<image>\n{INSTRUCTION_TEMPLATE.format(instruction=flattened_sample['instruction'])}"
                    }
                )

        # target step
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{flattened_sample['steps'][-1]['reasoning'].strip()}</think>\n\n{flattened_sample['steps'][-1]['response'].strip()}"
            }
        )

    # prepare images
    images = flattened_sample["screenshots"]

    # number of `<image>` should match number of images
    num_images = sum(m['content'].count("<image>") for m in messages)
    if num_images != len(images):
        ipdb.set_trace()
        pass

    return {
        "messages": messages,
        "images": images,
    }



if __name__ == "__main__":
    base_dir = Path("/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/trajectories")
    input_dirs = [
        base_dir / "kimi",
        base_dir / "kimi_mj",
        base_dir / "kimi_spreadsheetbench",
        base_dir / "kimi_zenodo",
    ]
    save_dir = "train.jsonl"

    # load all samples from input_dirs
    all_jsonl_files = []
    for path in input_dirs:
        trajectory_paths = list(path.resolve().rglob("**/trajectory.json"))
        all_jsonl_files += trajectory_paths

        print(f"Found {len(trajectory_paths)} trajectories in {path}.")

    # format samples
    samples = []
    for filename in tqdm(all_jsonl_files):
        with open(filename, "r") as f:
            trajectory_json = json.load(f)

        screenshots = [str(filename.parent / "0-0.png")]  # initial screenshot
        for step in trajectory_json["steps"][0]["actions"]:
            screenshots.append(step["screenshot"])

        steps = [
            {
                "reasoning": step["raw_reasoning"],
                "response": step["raw_response"],
            }
            for step in trajectory_json["steps"][0]["actions"]
        ]

        sample = {
            "instruction": trajectory_json["goal"],
            "screenshots": screenshots,
            "steps": steps
        }
        samples.append(sample)

    # format into ShareGPT
    all_sft_samples = []
    for sample in tqdm(samples, desc="ShareGPT formatting"):
        all_sft_samples += sample_to_sft_samples(sample)

    ipdb.set_trace()
    pass

    print(f"Number of unique trajectories: {len(samples)}")
    print(f"Number of SFT samples: {len(all_sft_samples)}")

    with jsonlines.open(save_dir, "w") as f:
        f.write_all(all_sft_samples)

    print(f"Saved {len(all_sft_samples)} SFT samples to {save_dir}.")


