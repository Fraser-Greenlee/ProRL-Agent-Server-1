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

    def generate_goal_with_persona(self, screenshot: bytes, persona: Dict, previous_intents: List[str],
                                   previous_goals: List[str],) -> Tuple[str, str]:
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

    def generate_action(self, screenshot: bytes, goal: str, previous_thoughts: List[str],
                        previous_actions: List[str]) -> Tuple[str, Dict]:
        messages = self.prepare_generate_action_messages(
            screenshot, goal, previous_thoughts, previous_actions
        )

        thought, action_dict, num_generation = None, None, 0
        while num_generation < self.max_retry_for_action_generation:
            responses, reasons = self.prompt_vlm_with_reason(
                messages, n=1, temperature=0.6, top_p=0.9, max_tokens=8192,
            )
            response, reason = responses[0], reasons[0]

            thought, action_dict = self.parse_thought_and_action_dict(response)
            if reason == "stop" and thought is not None and action_dict is not None:
                break

            num_generation += 1

        if num_generation == self.max_retry_for_goal_generation:
            raise OpenAIError("Goal Generation was not successful.")

        ipdb.set_trace()
        pass

        return thought, action_dict

    def evaluate_action(self, prev_screenshot: bytes, curr_screenshot: bytes, action_dict: Dict) -> str:
        messages = self.prepare_evaluate_action_messages(
            prev_screenshot, curr_screenshot, action_dict
        )

        ipdb.set_trace()
        pass

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
    def parse_thought_and_action_dict(generation: str) -> Tuple:
        generation = generation.split("</think>")[-1]

        # 1. Extract Thought
        thought_match = re.search(r'Thought:(.*?)(?=Action:|$)', generation, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "No reasoning provided."

        # 2. Extract JSON Action
        # This regex looks for the JSON structure starting after "Action:"
        json_match = re.search(r'Action:\s*(\{.*\})', generation, re.DOTALL | re.IGNORECASE)

        if json_match:
            json_str = json_match.group(1).strip()
            try:
                action_dict = json.loads(json_str)
                OpenAIController.validate_action(action_dict)
            except json.JSONDecodeError as e:
                logger.debug(f"Error in json decoding: {e}")
                thought, action_dict = None, None
            except ValueError as e:
                logger.debug(f"Error in parse_thought_and_action_dict: {e}")
                thought, action_dict = None, None
        else:
            thought, action_dict = None, None

        return thought, action_dict

    @staticmethod
    def validate_action(action_dict: dict):
        """
        Validates that the action_dict adheres to the action schema.
        """
        # todo take bbox_list as additional argument, and validate if the selected object_idx bbox exists in the image

        # 1. Check basic structure
        if not isinstance(action_dict, dict):
            raise ValueError("Action must be a JSON object (dictionary).")

        if "action" not in action_dict:
            raise ValueError("Missing required key: 'action'.")

        action_type = action_dict["action"]
        valid_actions = {
            "click", "double_click", "right_click",
            "scroll", "type", "hotkey", "wait", "done"
        }

        if action_type not in valid_actions:
            raise ValueError(f"Unknown action type: '{action_type}'. Must be one of {valid_actions}")

        # Ensure parameters dict exists (default to empty if missing for wait/done)
        parameters = action_dict.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a dictionary.")

        # 2. Validate specifics based on action type

        # --- Case A: Pointing Actions (Click variants) ---
        if action_type in ["click", "double_click", "right_click"]:
            if "object_idx" not in parameters:
                raise ValueError(f"Action '{action_type}' requires parameter 'object_idx'.")

            if not isinstance(parameters["object_idx"], int):
                raise ValueError(f"'{action_type}' parameter 'object_idx' must be an integer.")

        # --- Case B: Scroll ---
        elif action_type == "scroll":
            required_keys = ["object_idx", "amount_to_scroll"]
            for key in required_keys:
                if key not in parameters:
                    raise ValueError(f"Action 'scroll' requires parameter '{key}'.")
                if not isinstance(parameters[key], int):
                    raise ValueError(f"'scroll' parameter '{key}' must be an integer.")

        # --- Case C: Keyboard Actions (Type / Hotkey) ---
        elif action_type in ["type", "hotkey"]:
            if "type_list" not in parameters:
                raise ValueError(f"Action '{action_type}' requires parameter 'type_list'.")

            type_list = parameters["type_list"]

            # Check if it is a list
            if not isinstance(type_list, list):
                raise ValueError(f"'{action_type}' parameter 'type_list' must be a list.")

            # Check if all items in list are strings
            if not all(isinstance(item, str) for item in type_list):
                raise ValueError(f"'{action_type}' parameter 'type_list' must contain only strings.")

        # --- Case D: Wait / Done ---
        elif action_type in ["wait", "done"]:
            # These actions require no parameters, so we can ignore extra keys or ensure it's empty.
            # Strict version: ensure no parameters are passed?
            # Usually it's safer to just ignore extra parameters.
            pass

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

    @staticmethod
    def prepare_generate_action_messages(screenshot: bytes | Image.Image, goal: str, prev_thoughts: List[str],
                                         prev_actions: List[str]) -> List:
        # 1. Format History (Pairing Intents with Actions)
        if prev_actions:
            history_str = "\n".join([f"- Thought: {t} | Action: {a}" for t, a in zip(prev_thoughts, prev_actions)])
            # history_str = "\n".join([f"- Action: {a}" for t, a in zip(prev_thoughts, prev_actions)])
        else:
            history_str = "None (Start of action sequence for the current goal)"

        instruction_prompt = (
            f"You are an AI agent operating a computer. Your task is to generate a SINGLE JSON action object "
            f"to move closer to the current goal.\n\n"

            f"### CURRENT GOAL\n"
            f"\"{goal}\"\n\n"

            f"### HISTORY (Recent thoughts and actions)\n"
            f"{history_str}\n\n"

            f"### INSTRUCTIONS\n"
            f"1. **Analyze the Screenshot**: Look for the specific UI elements needed for your goal. Identify the index number (object_idx) on the element.\n"
            f"2. **Reason First**: Before generating the JSON, explicitly state your immediate intent (e.g., 'The File menu is at index 5, I need to click it').\n"
            f"3. **Loop Detection**: Review the History. If you see repeated actions (e.g., clicking one index multiple times) with the same intent, "
            f"you are stuck. Change your strategy (e.g., try scrolling, searching, or using a hotkey).\n"
            f"4. **Reference to the UI element**: Sometimes, the goal may ask you to interact with an UI element that is not associated with a bounding box. In "
            f"this case, do not hallucinate the index of a non-existing bounding box around that UI-element. Instead, try alternative approaches (e.g., the goal "
            f"is to close an app and the close button is not associated with a bounding box, try closing it with a HotKey action (that closes the current window) "
            f"rather than attempting to push the non-existing bounding box.\n"
            f"5. **How to Use History**: Do not use the history actions to get the relevant object idx. The index of the object to interact must be selected from "
            f"the current screenshot, not the action history.\n"
            f"6. **Output Format**: You must output two lines:\n"
            f"   - A 'Thought' line explaining your move.\n"
            f"   - An 'Action' line containing ONLY the valid JSON object.\n\n"

            f"### ACTION SPACE & EXAMPLES\n"
            f"Choose exactly one of the following actions. object_idx is the integer index of the UI element in the "
            f"screenshot. The index is located at the **bottom-left** of each bounding box; do not conflate the index "
            f"with e.g. upper-left side index.\n\n"

            f"**1. Click**\n"
            f"Use to click on an element\n"
            f"Example:\n"
            f"Action: {{\"action\": \"click\", \"parameters\": {{\"object_idx\": 12}}}}\n\n"

            f"**2. Double Click**"
            f"Use to double-click on an element.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"double_click\", \"parameters\": {{\"object_idx\": 0}}}}\n\n"

            f"**3. Right Click**\n"
            f"Use to right-click on top of an element.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"right_click\", \"parameters\": {{\"object_idx\": 50}}}}\n\n"

            f"**4. Scroll**\n"
            f"Use if the target is off-screen. 'object_idx' is the region to scroll. 'amount_to_scroll' "
            f"is positive (down) or negative (up).\n"
            f"Example:\n"
            f"Action: {{\"action\": \"scroll\", \"parameters\": {{\"object_idx\": 3, \"amount_to_scroll\": 2}}}}\n\n"

            f"**5. Type**\n"
            f"Use for entering text. 'type_list' is a list of characters or special keys.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"type\", \"parameters\": {{\"type_list\": [\"hello\", \"enter\"]}}}}\n\n"

            f"**6. Hotkey**\n"
            f"Use for keyboard shortcuts.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"hotkey\", \"parameters\": {{\"type_list\": [\"ctrl\", \"c\"]}}}}\n\n"

            f"**7. Wait**\n"
            f"Use if the screen is loading or processing.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"wait\"}}\n\n"

            f"**8. Done**\n"
            f"Use ONLY when the specific goal described above is fully completed.\n"
            f"Example:\n"
            f"Action: {{\"action\": \"done\"}}\n\n"

            f"### YOUR RESPONSE\n"
            f"Think with effort and verify your decision. Your final response should be formatted as follows:\n"
            f"Thought: [A sentence explaining your action choice]\n"
            f"Action: [JSON object]"
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
    def prepare_evaluate_action_messages(prev_screenshot: bytes | Image.Image, curr_screenshot: bytes | Image.Image,
                                         action_dict: Dict) -> List:
        # todo
        pass

