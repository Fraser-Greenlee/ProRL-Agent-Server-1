import json
import re
from argparse import Namespace
from typing import Dict, List, Tuple

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




        # instruction_prompt = (
        #     f"You are an AI agent exploring a desktop environment. Your task is to generate a SINGLE, realistic, and "
        #     f"specific next goal to pursue, given the initial setup of the OS and the current screenshot.\n\n"
        #
        #     f"### CONTEXT\n"
        #     f"- **OS Setup (Initial State)**:\n"
        #     f"{json.dumps(osworld_config, indent=4)}\n\n"
        #
        #     f"  *OS Setup Schema Reference*:\n"
        #     f"  The setup above defines the environment initialization using the following types:\n"
        #     f"  1. `download`: Files downloaded to disk (`path` is the path in the OS the file is downloaded).\n"
        #     f"  2. `open`: Files opened at startup. This implies specific apps are active (e.g., .xlsx opens in "
        #     f"LibreOffice Calc, .docx in Writer, images in GIMP/Viewer, code in VSCode, email in Thunderbird).\n"
        #     f"  3. `launch`: Launch apps with specific commands (e.g., `google-chrome --remote-debugging-port`, "
        #     f"`nautilus` file manager).\n"
        #     f"  4. `chrome_open_tabs`: Open the specified URLs as Chrome tabs.\n"
        #     f"  5. `chrome_close_tabs`: Close the specified URLS in Chrome.\n"
        #     f"  6. `update_browse_history`: Populates browser history (useful for 'revisiting' sites).\n"
        #     f"  7. `activate_window`: Put focus on the specific window initially.\n"
        #     f"  8. `execute`: Run shell commands (e.g., `mkdir` for creating directories).\n"
        #     f"  The setup will be executed in a sequential manner.\n"
        #     f"  *Note: There may be new types other than the ones described above. If you see unknown types, make an "
        #     f"educated guess based on the parameters.*\n\n"
        #
        #     f"- **History (Recent goals attempted)**:\n"
        #     f"{history_str}\n\n"
        #
        #     f"### INSTRUCTIONS\n"
        #     f"- **Visual Grounding**: The goal must be actionable based on the *visible* UI elements (icons, open "
        #     f"windows, menus) in the current screenshot.\n"
        #     f"- **Config Grounding**: Try to incorporate the files, apps, and settings that were set up in the OS setup, "
        #     f"especially when they were not used in previous goals.\n"
        #     f"- **Notes on Config**: Some setups (e.g., specific downloads or background processes) may not be "
        #     f"immediately visible in the screenshot. You may need to navigate (e.g., open File Manager) to find them. "
        #     f"Also, note that the OS Setup defines the *initial* state. Content in files or open windows may have "
        #     f"changed due to actions listed in the **History**.\n"
        #     f"- **Loop Detection**: Review the 'History' list. If the same goal appears multiple times recently, "
        #     f"you are likely stuck in a loop or failing to execute the action. **Do not generate the same goal again.** "
        #     f"Instead, backtrack or switch to a different task entirely.\n"
        #     f"- **Specificity**: Avoid vague goals, focus on specific, atomic goals. \n"
        #     f"   - BAD: 'Browse the internet' or 'Write something'.\n"
        #     f"   - GOOD: 'Open a web browser and navigate to wikipedia.org' or 'Save the currently open spreadsheet to "
        #     f"the desktop.'\n"
        #     f"- **Continuity**: The goal must naturally follow the history. If you just opened a terminal, the next "
        #     f"goal should be to run a specific command, not to immediately close it or doing something completely "
        #     f"unrelated. If you are not given with any history, try to make the best use of the existing content in "
        #     f"the screenshot and the config, rather than closing all windows.\n"
        #     f"- **Simplicity**: The goal should be able to be accomplished in at most 5 actions. The actions here "
        #     f"would include atomic interactions such as click, scroll, and typing.\n"
        #     f"- **Interaction Consideration**: The goal you write will be used as input to the action model. The "
        #     f"action space of the action model consists of following primitive actions - click, double-click, "
        #     f"right-click, scroll, type, and hotkey. If you are to specifically describe these actions in your goal, "
        #     f"make sure to not to conflate the words describing each action. Particularly, you need to be careful not "
        #     f"to conflate click, double-click, right-click. For example, it's better to say `double-click` a file on a "
        #     f"desktop to open it, rather than just `click` it (because clicking it wouldn't open a file on a "
        #     f"desktop).\n"
        #
        #     f"Your final response should be formatted as follows:\n"
        #     f"Intent: [A brief sentence - What is the long-term plan and what step are we on?]\n"
        #     f"New Goal: [your new goal]"
        # )

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

