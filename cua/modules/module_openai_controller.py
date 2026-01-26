import json
import re
from argparse import Namespace
from typing import Dict, List, Tuple

import ipdb
from PIL import Image

from openai import OpenAI, OpenAIError

from cua.modules.module_parser_controller import ParserController
from cua.modules.util import build_messages, bytes_to_image
from openhands.core.logger import openhands_logger
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime

# Create a child logger
logger = openhands_logger.getChild('openai_wrapper')


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
            self, messages: List, n: int = 1, temperature: float = 0.7, max_tokens: int = 8192
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
            timeout=int(600),
        )

        responses = [choice.message.content for choice in chat_response.choices]
        reasons = [choice.finish_reason for choice in chat_response.choices]

        return responses, reasons

    def generate_goal_with_persona(self, screenshot: bytes, persona: Dict, previous_intents: List[str],
                                   previous_goals: List[str],) -> Tuple[str, str]:
        # todo similar to SyntheticDataGenerator.generate_goal, generate high-level sub-goal to pick random actions
        messages = self.prepare_generate_goal_messages(
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

    def generate_cursor_moving_action(self, screenshot_with_cursor: bytes, goal: str, current_cursor_x: int,
                                      current_cursor_y: int, screenshot_len_x: int, screenshot_len_y: int) -> Tuple[int, int]:
        cursor_moving_messages = self.prepare_cursor_moving_messages(
            screenshot_with_cursor, goal, current_cursor_x, current_cursor_y, screenshot_len_x, screenshot_len_y,
        )
        responses, reasons = self.prompt_vlm_with_reason(
            cursor_moving_messages, n=1, temperature=0.7, max_tokens=8192,
        )
        response, reason = responses[0], reasons[0]

        # parse cursor x, cursor y
        generation = response.split("</think>")[-1]
        pattern = r"x\s*:\s*(-?\d+)\s+y\s*:\s*(-?\d+)"
        match = re.search(pattern, generation, re.IGNORECASE)

        if match:
            move_x, move_y = int(match.group(1)), int(match.group(2))
        else:
            move_x, move_y = -1, -1

        print(generation)

        return move_x, move_y

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
    def prepare_generate_goal_messages(screenshot: bytes | Image.Image, persona: Dict, prev_intents: List[str],
                                       prev_goals: List[str]) -> List:
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
            f"desktop).\n\n"

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
    def prepare_cursor_moving_messages(screenshot: bytes | Image.Image, goal: str,
                                       current_cursor_x: int, current_cursor_y: int,
                                       screenshot_len_x: int, screenshot_len_y: int) -> List:
        # instruction_prompt = (
        #     f"You are an AI agent controlling a mouse cursor. Your task is to calculate the precise relative movement "
        #     f"needed to move the cursor tip from its current position to the target UI element required by the goal.\n\n"
        #
        #     f"### CONTEXT\n"
        #     f"1. **Screen Resolution**: {screenshot_len_x} pixels wide (x), {screenshot_len_y} pixels tall (y).\n"
        #     f"2. **Coordinate System**: (0, 0) is the top-left corner. X increases to the right, Y increases downwards.\n"
        #     f"3. **Current Cursor Position**: (x={current_cursor_x}, y={current_cursor_y}), the position is surrounded by "
        #     f"a bounding box in the image to help its location.\n"
        #     f"4. **Current Goal**: \"{goal}\"\n\n"
        #
        #     f"### INSTRUCTIONS\n"
        #     f"1. **Identify Target**: Locate the center of the UI element that you have to interact with in the current "
        #     f"screenshot, to achieve the given goal.\n"
        #     f"2. **Estimate Coordinates**: Estimate the absolute (x, y) pixel coordinates of that target's center.\n"
        #     f"3. **Calculate Delta**: Calculate the relative distance to move.\n"
        #     f"   - Move X = Target X - Current Cursor X\n"
        #     f"   - Move Y = Target Y - Current Cursor Y\n"
        #     f"   - Example: If cursor is at 100 and target is at 150, Move X is 50. If target is at 50, Move X is -50.\n"
        #     f"4. If you do not need to move the cursor to achieve the goal (e.g., you want to type on an element and "
        #     f"the element is already on focus, or the cursor is already on the button you want to press), just output "
        #     f"x: 0, y: 0.\n"
        #     f"5 You do not have to locate the cursor on the exact center point; If the cursor is in the "
        #     f"enough position to interact with the target element, output x: 0, y: 0.\n"
        #     f"6. **Output Format**: Return the result as a JSON object.\n\n"
        #
        #     f"### EXAMPLES\n"
        #     f"**Example 1**\n"
        #     f"Goal: 'Click the Start Menu icon (bottom left)'\n"
        #     f"Current Cursor: (1000, 500)\n"
        #     f"Thought: The Start icon is at roughly (20, 1060). \n"
        #     f"   Move X: 20 - 1000 = -980\n"
        #     f"   Move Y: 1060 - 500 = 560\n"
        #     f"Your response: x: -980, y: 560\n\n"
        #
        #     f"**Example 2**\n"
        #     f"Goal: 'Close the window (X icon top right)'\n"
        #     f"Current Cursor: (500, 500)\n"
        #     f"Thought: The X icon is at roughly (1900, 20). \n"
        #     f"   Move X: 1900 - 500 = 1400\n"
        #     f"   Move Y: 20 - 500 = -480\n"
        #     f"Your response: x: 1400, y: -480\n\n"
        #
        #     f"Your final response should be formatted as follows:\n"
        #     f"x: [amount to move in x axis]\n"
        #     f"y: [amount to move in y axis]"
        # )

        instruction_prompt = (
            f"You are an AI agent controlling a mouse cursor. Your task is to calculate the precise relative movement "
            f"needed to move the cursor tip from its current position to the target UI element required by the goal.\n\n"

            f"### CONTEXT\n"
            f"1. **Screen Resolution**: {screenshot_len_x} pixels wide (x), {screenshot_len_y} pixels tall (y).\n"
            f"2. **Coordinate System**: (0, 0) is the top-left corner. X increases to the right, Y increases downwards.\n"
            f"3. **Current Cursor Position**: (x={current_cursor_x}, y={current_cursor_y}), the position is surrounded by "
            f"a bounding box in the image to help its location.\n"
            f"4. **Current Goal**: \"{goal}\"\n\n"

            f"### INSTRUCTIONS\n"
            f"1. **Identify Target**: Locate the center of the UI element that you have to interact with in the current "
            f"screenshot, to achieve the given goal.\n"
            f"2. **Estimate Distance**: Estimate the pixel distance in x, y axis between the target element's center and the cursor bounding box.\n"
            f"3. **Determine X, Y**: Based on the estimated distance, determine the movement of cursor in x, y axis. "
            f"If the cursor is currently located left of the target, you should move the cursor to the right, thus the "
            f"movement needed is exactly the pixel distance. If the cursor is located right of the target, you should move "
            f"the cursor to the left, thus the movement needed is exactly pixel distance * -1."
            f"4. If you do not need to move the cursor to achieve the goal (e.g., you want to type on an element and "
            f"the element is already on focus, or the cursor is already on the button you want to press), just output "
            f"x: 0, y: 0.\n"
            f"5. You do not have to locate the cursor on the exact center point; If the cursor is in the "
            f"enough position to interact with the target element, output x: 0, y: 0.\n"
            f"6. **Output Format**: Return the result as a JSON object.\n\n"

            f"### EXAMPLES\n"
            f"**Example 1**\n"
            f"Goal: 'Click the Start Menu icon (bottom left cursor)'\n"
            f"Current Cursor: (1000, 500)\n"
            f"The Start icon is about 980 pixel away from the cursor in x-axis, 560 pixel away in y-axis.\n"
            f"   Move X: The cursor is located right of the start icon, thus we multiply -1 * 980 = -980\n"
            f"   Move Y: The cursor is located above the start icon, thus Move Y = 560"
            f"Your response: x: -980, y: 560\n\n"

            f"**Example 2**\n"
            f"Goal: 'Close the window by clicking top-right X button'\n"
            f"Current Cursor: (500, 500)\n"
            f"Thought: The X icon is at roughly 1400 pixel away from the cursor in x-axis, 485 pixel away in y-axis. \n"
            f"   Move X: The cursor is located left of the X icon, thus Move X = 1400\n"
            f"   Move Y: The cursor is located below the X icon, thus Move Y = -1 * 485 = -485\n"
            f"Your response: x: 1400, y: -485\n\n"

            f"Your final response should be formatted as follows:\n"
            f"x: [amount to move in x axis]\n"
            f"y: [amount to move in y axis]"
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

