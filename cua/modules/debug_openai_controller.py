import ast
import json
import re
from argparse import Namespace
from typing import Dict, List, Tuple, Optional

import ipdb
from PIL import Image

from openai import OpenAI, OpenAIError

from cua.modules.util import build_messages, bytes_to_image
from openhands.core.logger import openhands_logger

# Create a child logger
logger = openhands_logger.getChild('openai_controller')


class OpenAIController:
    """
    Wrapper class that supports all interactions with remote vLLM server.
    """
    def __init__(self, args: Namespace):
        self.client = OpenAI(
            base_url=f"http://{args.explorer_node}:8000/v1",
            api_key="gen",
        )
        self.model_name = args.model_name
        self.max_retry_for_goal_generation = args.max_retry_for_goal_generation
        self.max_retry_for_action_generation = args.max_retry_for_action_generation

        if args.min_pixels != -1 and args.max_pixels != -1:
            self.mm_processor_kwargs = {
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        else:
            self.mm_processor_kwargs = None

    def prompt_vlm_with_reason(
            self, messages: List, n: int = 1, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8192
    ) -> Tuple[List, List]:
        """
        Helper function to prompt VLM with messages.
        Use with try...except phrase to process timeout error.
        """
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={
                "mm_processor_kwargs": None,
            },
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=int(600),
        )

        responses = [choice.message.content for choice in chat_response.choices]
        reasons = [choice.finish_reason for choice in chat_response.choices]

        return responses, reasons

    def generate_subgoal(
            self, screenshot: bytes | Image.Image, goal: str, prev_subgoal_intents: List[str],
            prev_subgoals: List[str], prev_actor_infos: List[str],
    ) -> Tuple[str, str]:
        messages = self.prepare_generate_subgoal_messages(
            screenshot, goal, prev_subgoal_intents, prev_subgoals, prev_actor_infos,
        )

        intent, goal, num_generation = None, None, 0
        while num_generation < self.max_retry_for_goal_generation:
            responses, reasons = self.prompt_vlm_with_reason(
                messages, n=1, temperature=0.7, max_tokens=8192,
            )
            response, reason = responses[0], reasons[0]

            intent, goal = self.parse_intent_and_goal(response)
            if reason == "stop" and intent is not None and goal is not None:
                break

            num_generation += 1

        if num_generation == self.max_retry_for_goal_generation:
            raise OpenAIError("Goal Generation was not successful.")

        return intent, goal

    def generate_goal_with_long_horizon(
            self, screenshot: bytes | Image.Image, osworld_config: List[Dict], example_goals: List[str],
            prev_requirements: List[Tuple[str, str]],
    ) -> Tuple[str, List[str]]:
        """
        Generate OSWorld-aligned goals by leveraging examples from AgentNet.
        Output consists of (1) the new goal, (2) requirements in the environment that must be met for the goal to be
        achievable.
        The workflow is like this:
        - We generate goal + requirement
        - We verify each of the requirement, and obtain the verification result.
        - If the requirement is not satisfied, we generate another goal - this time, we concatenate the requirement +
        condition + verification result so that the model does not make same mistake.
        """
        messages = self.prepare_generate_goal_with_long_horizon(
            screenshot, osworld_config, example_goals, prev_requirements
        )

        requirements, goal, num_generation = None, None, 0
        while num_generation < self.max_retry_for_goal_generation:
            responses, reasons = self.prompt_vlm_with_reason(
                messages, n=1, temperature=1.0, max_tokens=8192,
            )
            response, reason = responses[0], reasons[0]

            goal, requirements = self.parse_with_long_horizon(response)
            if reason == "stop" and goal is not None and requirements is not None:
                break

            num_generation += 1

        if num_generation == self.max_retry_for_goal_generation:
            raise OpenAIError("Goal Generation was not successful.")

        return goal, requirements

    def generate_goal_with_osworld_config(
            self, screenshot: bytes | Image.Image, osworld_config: List[Dict], prev_intents: List[str],
            prev_goals: List[str], prev_actor_infos: List[str]
    ) -> Tuple[str, str]:
        """
        Generate new goal based on osworld setup (to promote better use of files & settings in the setup).
        """
        # todo get the previous UI-TARS last thought as prev_actor_info
        messages = self.prepare_generate_goal_with_osworld_config_messages(
            screenshot, osworld_config, prev_intents, prev_goals, prev_actor_infos
        )

        intent, goal, num_generation = None, None, 0
        while num_generation < self.max_retry_for_goal_generation:
            responses, reasons = self.prompt_vlm_with_reason(
                messages, n=1, temperature=1.0, max_tokens=8192,
            )
            response, reason = responses[0], reasons[0]

            intent, goal = self.parse_intent_and_goal(response)
            if reason == "stop" and intent is not None and goal is not None:
                break

            num_generation += 1

        if num_generation == self.max_retry_for_goal_generation:
            raise OpenAIError("Goal Generation was not successful.")

        return intent, goal

    def generate_goal_with_persona(
            self, screenshot: bytes, persona: Dict, previous_intents: List[str], previous_goals: List[str]
    ) -> Tuple[str, str]:
        """
        Generate new goal based on a sampled persona.
        """
        messages = self.prepare_generate_goal_with_persona_messages(
            screenshot, persona, previous_intents, previous_goals,
        )

        intent, goal, num_generation = None, None, 0
        while num_generation < self.max_retry_for_goal_generation:
            responses, reasons = self.prompt_vlm_with_reason(
                messages, n=1, temperature=0.7, max_tokens=8192,
            )
            response, reason = responses[0], reasons[0]

            intent, goal = self.parse_intent_and_goal(response)
            if reason == "stop" and intent is not None and goal is not None:
                break

            num_generation += 1

        if num_generation == self.max_retry_for_goal_generation:
            raise OpenAIError("Goal Generation was not successful.")

        return intent, goal

    @staticmethod
    def parse_with_long_horizon(generation: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Parses the model output to extract the 'New Goal' string and the 'Requirements' list.
        Handles potential chain-of-thought outputs (</think>) and various list formats.
        """
        # 1. Strip chain-of-thought if present
        if "</think>" in generation:
            generation = generation.split("</think>")[-1]

        # 2. Check for required markers
        if "New Goal: " not in generation or "Requirements: " not in generation:
            return None, None

        try:
            # 3. Split content
            # Remove everything before "New Goal:" to handle preamble noise
            content_start = generation.index("New Goal: ")
            clean_generation = generation[content_start:]

            # Split into parts
            part1 = clean_generation.split("Requirements: ", 1)
            if len(part1) != 2:
                return None, None

            raw_goal_str, raw_reqs_str = part1

            # 4. Clean Goal
            goal = raw_goal_str.replace("New Goal: ", "").strip()

            # 5. Clean and Parse Requirements
            # The model might output: ["req1", "req2"]
            # Or sometimes code blocks: ```json ["req1"] ```
            raw_reqs_str = raw_reqs_str.strip()

            # Remove markdown code fences if present
            if raw_reqs_str.startswith("```"):
                lines = raw_reqs_str.splitlines()
                # Strip first line (```json) and last line (```)
                if len(lines) >= 3:
                    raw_reqs_str = "\n".join(lines[1:-1])
                else:
                    raw_reqs_str = raw_reqs_str.strip("`").replace("json", "")

            # Attempt to parse list
            try:
                requirements = json.loads(raw_reqs_str)
            except json.JSONDecodeError:
                try:
                    # Fallback for Python-style lists (e.g. single quotes)
                    requirements = ast.literal_eval(raw_reqs_str)
                except (ValueError, SyntaxError):
                    # Fallback: simple line splitting if list parsing completely fails
                    # This handles cases where the model outputs a bulleted list instead of a JSON list
                    requirements = [line.strip("- *") for line in raw_reqs_str.splitlines() if line.strip()]

            # 6. Validation
            if not isinstance(requirements, list):
                return None, None

            # Filter out too long goals/requirements as a safety check
            if len(goal) >= 1000:
                return None, None

            # Ensure all requirements are strings
            requirements = [str(r) for r in requirements if isinstance(r, str) or isinstance(r, (int, float))]

            return goal, requirements

        except Exception as e:
            # Catch-all for unexpected parsing errors
            # print(f"Parsing error: {e}") # Optional logging
            return None, None

    @staticmethod
    def parse_intent_and_goal(generation: str) -> Tuple:
        generation = generation.split("</think>")[-1]

        if "Intent: " not in generation or "New Goal: " not in generation:
            return None, None

        generation = generation.split("Intent: ")[-1].strip()
        intent, goal = generation.split("New Goal: ", 1)

        # filter out too long intent or goal
        if len(intent) >= 1000 or len(goal) >= 1000:
            return None, None

        return intent.strip(), goal.strip()

    @staticmethod
    def prepare_generate_subgoal_messages(
            screenshot: bytes | Image.Image, goal: str, prev_subgoal_intents: List[str], prev_subgoals: List[str],
            prev_actor_infos: List[str],
    ) -> List:
        if prev_subgoal_intents:
            assert len(prev_subgoal_intents) == len(prev_subgoals) == len(prev_actor_infos)
            history_str = "\n".join([f"- Intent: {i} | Sub-Goal: {g} | Result: {r}"
                                     for i, g, r in zip(prev_subgoal_intents, prev_subgoals, prev_actor_infos)])
        else:
            history_str = "None (Session Start)"

        instruction_prompt = (
            f"You are an AI agent exploring a desktop environment. Your task is to generate a SINGLE, realistic, and "
            f"specific next sub-goal to pursue, given the high-level goal of the user and the current screenshot.\n\n"

            f"### CURRENT CONTEXT\n"
            f"1. **High-level Goal**:\n"
            f"{goal}\n\n"
            f"2. **History (Recent goals attempted)**:\n"
            f"{history_str}\n\n"

            f"### INSTRUCTIONS\n"
            f"Analyze the provided screenshot and the context above to determine the next logical step. Adhere to "
            f"the following rules:\n"
            f"1. **Goal Direction**: The sub-goal should direct towards achieving the high-level goal. Make sure to "
            f"take into account the current visual state shown on the screenshot, and reason where the current stage "
            f"is based on the history of previous goals attempted.\n"
            f"2. **Visual Grounding**: The sub-goal must be actionable based on the *visible* UI elements (icons, open "
            f"windows, menus) in the current screenshot. Do not assume apps are installed or files exist (unless you "
            f"confirmed that they exist in the directory) without seeing them.\n"
            f"3. **Loop Detection**: Review the 'History' list. If the same goal appears multiple times recently, "
            f"the actor model that executes your sub-goals might be stuck in a loop. **Do not generate the same goal "
            f"again.** Try to think about why the sub-goal is not being achieved, and try to make the sub-goal easier "
            f"for the actor model to execute (e.g., specifying where the button to click is).\n"
            f"4. **Specificity**: Avoid vague goals, and try to be specific.\n"
            f"   - BAD: 'Browse the internet' or 'Write something'.\n"
            f"   - GOOD: 'Open a web browser and navigate to wikipedia.org', 'Save the currently open spreadsheet to "
            f"the desktop.', 'Copy the number of animals in the current text to the thunderbird email window.'\n"
            f"Try not to be overly atomic - e.g., you can ask for more than just clicking a single button. Think of "
            f"the sub-goals as at most 5-10 intermediate steps you need to take in order to achieve the goal. However, "
            f"if the history shows that multiple goals are duplicated and you are likely stuck in a loop, you can "
            f"think of a specific sub-goal to get out of that loop (e.g., by specifically instructing to click a certain "
            f"button)."
            f"5. If high-level goal is already achieved according to the history so far, write `DONE` as your new "
            f"goal.\n"
            f"6. If you find that the high-level goal is impossible to achieve due to its discrepancy with the current "
            f"desktop environment (e.g., the goal is referring to non-existing file, etc), write `IMPOSSIBLE` as your "
            f"new goal.\n"

            f"Your final response should be formatted as follows:\n"
            f"Intent: [A brief sentence - What is the long-term plan and what step are we on?]\n"
            f"New Goal: [your new goal]"
        )

        if isinstance(screenshot, bytes):
            screenshot = bytes_to_image(screenshot)

        line = {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>{instruction_prompt}",
                },
            ],
            "images": [screenshot],
        }
        messages = build_messages(line, min_pixels=4 * 28 * 28, max_pixels=5120 * 28 * 28)
        return messages

    @staticmethod
    def prepare_generate_goal_with_long_horizon(
            screenshot: bytes | Image.Image, osworld_config: List[Dict], example_goals: List[str],
            prev_requirements: List[Tuple[str, str]],
    ) -> List:
        if prev_requirements:
            prev_requirements_str = "\n".join(f"- Condition: {c} | Verdict: {v}" for c, v in prev_requirements)
        else:
            prev_requirements_str = "None (Initial Attempt)"

        example_goals_str = "\n".join(f"- {g}" for g in example_goals)

        instruction_prompt = (
            f"You are an agent generating synthetic data for OS World environment. Given example instruction(s), your "
            f"task is to generate a realistic, similar-in-style, similar-in-complexity instruction suited for a new "
            f"environment.\n\n"

            f"### Information on New Environment\n"
            f"- **OS Setup (Configuration)**:\n"
            f"{json.dumps(osworld_config, indent=4)}\n"
            f"  *Schema Reference*: `download` (file on disk), `open` (app active), `launch` (command run), "
            f"`chrome_open_tabs` (browser tabs), `execute` (shell command).\n\n"

            f"- **Visual State (Screenshot)**:\n"
            f"  The provided screenshot represents the *current* initial state for this new task. "
            f"  **Crucially: If the Screenshot contradicts the OS Setup (e.g., config says a window is open, but the screen shows it closed), trust the Screenshot.**\n\n"

            f"- **Previous Requirements**:\n"
            f"{prev_requirements_str}\n\n"

            f"### Example Goals\n"
            f"{example_goals_str}\n\n"

            f"### INSTRUCTIONS\n"
            f"1. **Analyze Achievability**: Your goal must be achievable given the current screenshot and OS Setup. "
            f"You do not have to stick to what is in the current screenshot when creating the goal - you are "
            f"encouraged to leverage information not immediately present in the screenshot, but they must be present "
            f"in the current OS setup. If the OS setup is empty, assume that the OS is in a clean state, thus it "
            f"wouldn't make sense to refer to custom downloaded files in that case. Make sure not to hallucinate any "
            f"non-existing files that cannot be found in either the screenshot or OS setup.\n"
            f"2. **Learn from Failures**: Review the 'Previous Requirements' section carefully.\n"
            f"   - The 'Verdict' explains whether a specific requirement is met in the current environment.\n"
            f"   - **Do not** generate a goal that relies on a condition that has already been proven impossible. \n"
            f"   - Use the verdict to steer your new goal toward resources that *are* available or achievable.\n"
            f"3. **Style & Complexity**: Generate a goal similar in style and complexity to the 'Example Goals'. Do "
            f"not just copy them; adapt the logic to the files and apps visible in this environment.\n"
            f"4. **Define Requirements**: List the specific, **objective** pre-conditions on the environment, required "
            f"for your goal to make sense in the environment. The conditions should be in a question form.\n"
            f"   - Ensure requirements are binary (True/False) questions, and are standalone, i.e., can be answered "
            f"by inspecting the desktop environment without looking at other requirement questions.\n"
            f"   - Example: Does a file named `report.pdf` exist on the Desktop? (Good)\n"
            f"   - Example: Does the pdf file `report.pdf` on the Desktop look professional? (Bad - Subjective)\n\n"

            f"Your final response should be formatted as follows:\n"
            f"New Goal: [your new goal]\n"
            f"Requirements: [\"your requirement 1\", \"your requirement 2\", ... max 5 requirements]\n"
        )

        if isinstance(screenshot, bytes):
            screenshot = bytes_to_image(screenshot)

        line = {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>{instruction_prompt}",
                },
            ],
            "images": [screenshot],
        }
        messages = build_messages(line, min_pixels=4 * 28 * 28, max_pixels=5120 * 28 * 28)
        return messages

    @staticmethod
    def prepare_generate_goal_with_osworld_config_messages(
            screenshot: bytes | Image.Image, osworld_config: List[Dict], prev_intents: List[str], prev_goals: List[str],
            prev_actor_infos: List[str]
    ) -> List:
        # todo add notepad to the goal-setting agent so that it can write some information (keeps growing)

        if prev_goals:
            assert len(prev_goals) == len(prev_intents) == len(prev_actor_infos)
            history_str = "\n".join([f"- Intent: {i} | Goal: {g} | Result: {r}"
                                     for i, g, r in zip(prev_intents, prev_goals, prev_actor_infos)])
        else:
            history_str = "None (Session Start)"

        instruction_prompt = (
            f"You are an AI agent exploring a desktop environment. Your task is to generate a SINGLE, realistic, and "
            f"specific next goal to pursue, given the initial setup of the OS and the current screenshot.\n\n"

            f"### CONTEXT\n"
            f"- **OS Setup (Initial State)**:\n"
            f"{json.dumps(osworld_config, indent=4)}\n\n"

            f"  *OS Setup Schema Reference*:\n"
            f"  The setup above defines the environment initialization using the following types:\n"
            f"  1. `download`: Files downloaded to disk (`path` is the path in the OS the file is downloaded).\n"
            f"  2. `open`: Files opened at startup. This implies specific apps are active (e.g., .xlsx opens in "
            f"LibreOffice Calc, .docx in Writer, images in GIMP/Viewer, code in VSCode, email in Thunderbird).\n"
            f"  3. `launch`: Launch apps with specific commands (e.g., `google-chrome --remote-debugging-port`, "
            f"`nautilus` file manager).\n"
            f"  4. `chrome_open_tabs`: Open the specified URLs as Chrome tabs.\n"
            f"  5. `chrome_close_tabs`: Close the specified URLS in Chrome.\n"
            f"  6. `update_browse_history`: Populates browser history (useful for 'revisiting' sites).\n"
            f"  7. `activate_window`: Put focus on the specific window initially.\n"
            f"  8. `execute`: Run shell commands (e.g., `mkdir` for creating directories).\n"
            f"  The setup will be executed in a sequential manner.\n"
            f"  *Note: There may be new types other than the ones described above. If you see unknown types, make an "
            f"educated guess based on the parameters.*\n\n"

            f"- **History (Recent goals attempted)**:\n"
            f"{history_str}\n\n"

            f"### INSTRUCTIONS\n"
            f"- **Visual Grounding**: The goal must be actionable based on the *visible* UI elements (icons, open "
            f"windows, menus) in the current screenshot.\n"
            f"- **Config Grounding**: Try to incorporate the files, apps, and settings that were set up in the OS setup, "
            f"especially when they were not used in previous goals.\n"
            f"- **Notes on Config**: Some setups (e.g., specific downloads or background processes) may not be "
            f"immediately visible in the screenshot. You may need to navigate (e.g., open File Manager) to find them. "
            f"Also, note that the OS Setup defines the *initial* state. Content in files or open windows may have "
            f"changed due to actions listed in the **History**.\n"
            f"- **Loop Detection**: Review the 'History' list. If the same goal appears multiple times recently, "
            f"you are likely stuck in a loop or failing to execute the action. **Do not generate the same goal again.** "
            f"Instead, backtrack or switch to a different task entirely.\n"
            f"- **Specificity**: Avoid vague goals, focus on specific, atomic goals. \n"
            f"   - BAD: 'Browse the internet' or 'Write something'.\n"
            f"   - GOOD: 'Open a web browser and navigate to wikipedia.org' or 'Save the currently open spreadsheet to "
            f"the desktop.'\n"
            f"- **Continuity**: The goal must naturally follow the history. If you just opened a terminal, the next "
            f"goal should be to run a specific command, not to immediately close it or doing something completely "
            f"unrelated. If you are not given with any history, try to make the best use of the existing content in "
            f"the screenshot and the config, rather than closing all windows.\n"
            f"- **Interaction Consideration**: The goal you write will be used as input to the action model. The "
            f"action space of the action model consists of following primitive actions - click, double-click, "
            f"right-click, scroll, type, and hotkey. If you are to specifically describe these actions in your goal, "
            f"make sure to not to conflate the words describing each action. Particularly, you need to be careful not "
            f"to conflate click, double-click, right-click. For example, it's better to say `double-click` a file on a "
            f"desktop to open it, rather than just `click` it (because clicking it wouldn't open a file on a "
            f"desktop).\n"

            f"Your final response should be formatted as follows:\n"
            f"Intent: [A brief sentence - What is the long-term plan and what step are we on?]\n"
            f"New Goal: [your new goal]"
        )

        if isinstance(screenshot, bytes):
            screenshot = bytes_to_image(screenshot)

        line = {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>{instruction_prompt}",
                },
            ],
            "images": [screenshot],
        }
        messages = build_messages(line, min_pixels=4 * 28 * 28, max_pixels=5120 * 28 * 28)
        return messages

    @staticmethod
    def prepare_generate_goal_with_persona_messages(screenshot: bytes | Image.Image, persona: Dict,
                                                    prev_intents: List[str], prev_goals: List[str]) -> List:
        if prev_goals:
            history_str = "\n".join([f"- Intent: {i} | Goal: {g}" for i, g in zip(prev_intents, prev_goals)])
        else:
            history_str = "None (Session Start)"

        # Format persona details
        persona_str = "\n".join(f"- {key.capitalize()}: {value}" for key, value in persona.items())

        instruction_prompt = (
            f"You are an AI agent exploring a desktop environment. Your task is to generate a SINGLE, realistic, and "
            f"specific next goal to pursue, given a persona of the user and the current screenshot.\n\n"

            f"### CURRENT CONTEXT\n"
            f"1. **Persona**:\n{persona_str}\n"
            f"2. **History (Recent goals attempted)**:\n{history_str}\n\n"

            f"### INSTRUCTIONS\n"
            f"Analyze the provided screenshot and the context above to determine the next logical step. "
            f"Adhere to the following rules:\n"
            f"1. **Visual Grounding**: The goal must be actionable based on the *visible* UI elements (icons, open "
            f"windows, menus) in the current screenshot. Do not assume apps are installed or files exist (unless you "
            f"confirmed that they exist in the directory) without seeing them.\n"
            f"2. **Loop Detection**: Review the 'History' list. If the same goal appears multiple times recently, "
            f"you are likely stuck in a loop or failing to execute the action. **Do not generate the same goal again.** "
            f"Instead, backtrack or switch to a different task entirely.\n"
            f"3. **Specificity**: Avoid vague goals, focus on specific, atomic goals. \n"
            f"   - BAD: 'Browse the internet' or 'Write something'.\n"
            f"   - GOOD: 'Open a web browser and navigate to wikipedia.org' or 'Save the currently open spreadsheet to "
            f"the desktop.'\n"
            f"4. **Continuity**: The goal must naturally follow the history. If you just "
            f"opened a terminal, the next goal should be to run a specific command, not to immediately close it or doing "
            f"something completely unrelated. If you are not given with any history, try to make the best use of the "
            f"existing content in the screenshot, rather than closing all windows.\n"
            f"5. **Persona Alignment**: Choose a goal that this specific persona would likely do.\n"
            f"6. **Simplicity**: The goal should be able to be accomplished in at most 5 actions. The actions here "
            f"would include atomic interactions such as click, scroll, and typing.\n"
            f"7. **Interaction Consideration**: The goal you write will be used as input to the action model. The "
            f"action space of the action model consists of following primitive actions - click, double-click, "
            f"right-click, scroll, type, and hotkey. If you are to specifically describe these actions in your goal, "
            f"make sure to not to conflate the words describing each action. Particularly, you need to be careful not "
            f"to conflate click, double-click, right-click. For example, it's better to say `double-click` a file on a "
            f"desktop to open it, rather than just `click` it (because clicking it wouldn't open a file on a "
            f"desktop).\n"
            f"8. **Element Consideration**: In the screenshot, you will see many bounding boxes that denote "
            f"interactive elements. Verify that your action starts off one of those elements with bounding boxes, "
            f"not those without bounding boxes. For example, if the image shows a close-button without a bounding box "
            f"surrounding it, you should not set your goal to involve an interaction with that button.\n\n"

            f"Your final response should be formatted as follows:\n"
            f"Intent: [A brief sentence - What is the long-term plan and what step are we on?]\n"
            f"New Goal: [your new goal]"
        )

        if isinstance(screenshot, bytes):
            screenshot = bytes_to_image(screenshot)

        line = {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image>{instruction_prompt}",
                },
            ],
            "images": [screenshot],
        }
        messages = build_messages(line, min_pixels=4 * 28 * 28, max_pixels=5120 * 28 * 28)
        return messages

