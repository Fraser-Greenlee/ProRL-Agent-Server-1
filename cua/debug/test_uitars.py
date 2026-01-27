import ast
import io

import ipdb
import math
import base64
from io import BytesIO
from typing import List, Dict, Union, Optional, Any, Tuple
import re

from PIL import Image
from openai import OpenAI
from qwen_vl_utils import fetch_image


def image_to_bytes(image: Image.Image) -> bytes:
    # todo use util version
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


def image_to_base64(image: Image.Image) -> str:
    # todo use util version
    encoded_image_str = base64.b64encode(image_to_bytes(image)).decode("utf-8")
    return f"data:image;base64,{encoded_image_str}"


def process_image(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    # todo move to util
    if min_pixels != -1 and max_pixels != -1:
        image_dict = {
            "image": image,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
    else:
        image_dict = {
            "image": image,
        }
    return fetch_image(image_dict)


# NOTE Reference: https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/uitars_agent.py
class UITarsController:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://0.0.0.0:8000/v1",
            api_key="gen",
        )
        self.model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
        self.max_retry_for_action_generation = 5

    def prompt_vlm_with_reason(
            self, messages: List, n: int = 1, temperature: float = 0, max_tokens: int = 4096,
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

    @staticmethod
    def parse_thought_and_action(generation, model_input_width: int, model_input_height: int) -> Union[None, List[Dict[str, Any]]]:
        """
        Parse thought and action from UI-Tars generation.
        Reference: uitars_agent.parse_action_qwen2vl
        Returns:
            None if error in parsing
            List of Dictionary if successful, {
                "reflection": str,
                "thought": str,
                "action_type": str (e.g. click),
                "action_inputs": dict (e.g. {"start_box": "[0.089, 0.424]"}
                "generation": str (raw generation)
            }
        """
        try:
            assert "Action:" in generation, f"`Action:` does not exist in generation {generation}"

            # 0. prepare smart_resize_height, smart_resize_width
            # smart_resize_height, smart_resize_width = UITarsController.smart_resize(screen_height, screen_width)

            # 1. Parse thought, reflection (optional), action from generation
            generation = UITarsController.parse_point_coordinates(generation)

            generation = generation.strip()
            if generation.startswith("Thought:"):
                thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
            elif generation.startswith("Reflection:"):
                thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
            elif generation.startswith("Action_Summary:"):
                thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
            else:
                # there is no explicit 'Thought:' marker, extract everything before 'Action:' as the thought.
                thought_pattern = r"(.+?)(?=\s*Action:|$)"

            reflection, thought = None, None
            thought_match = re.search(thought_pattern, generation, re.DOTALL)
            if thought_match:
                if len(thought_match.groups()) == 1:
                    thought = thought_match.group(1).strip()
                elif len(thought_match.groups()) == 2:
                    reflection = thought_match.group(1).strip()
                    thought = thought_match.group(2).strip()
                else:
                    raise ValueError
            else:
                raise ValueError
            action_str = generation.split("Action:")[-1]

            # 2. Parse actions from action_str
            tmp_all_action = action_str.split("\n\n")
            all_action = []
            for action_str in tmp_all_action:
                # handle escape letter issues like type(content='Don't stop')
                match = re.search(r"type\(content='(.*?)'\)", action_str)
                if match:
                    # Extract the text inside the quotes (Group 1)
                    content = match.group(1)

                    # Process and reconstruct
                    action_str = UITarsController.escape_single_quotes(content)
                    action_str = f"type(content='{action_str}')"

                if not action_str.strip().endswith(")"):
                    action_str = action_str.strip() + ")"

                all_action.append(action_str)

            # 3. Parse actions in all_action
            parsed_actions = [UITarsController.parse_function_call(action.replace("\n", "\\n").lstrip())
                              for action in all_action]
            actions = []
            for action_instance, raw_str in zip(parsed_actions, all_action):
                if action_instance is None:
                    print(f"Failed to parse action with parse_function_call: {raw_str}")
                    continue

                action_type, params = action_instance["function"], action_instance["args"]

                # process params into action_inputs
                action_inputs = {}
                for param_name, param in params.items():
                    if param == "":
                        continue
                    param = param.lstrip()
                    action_inputs[param_name.strip()] = param

                    if "point" in param_name:
                        bbox_str = param  # e.g. "(595,25)"

                        # Remove parentheses and split the string by commas
                        numbers = bbox_str.replace("(", "").replace(")", "").split(",")

                        # Previous (Qwen2-VL): Model outputs relative coordinates (i.e., percentage), scaled by 1000
                        # float_numbers = [float(num) / 1000 for num in numbers]

                        # Current (Qwen2.5-VL): Model outputs absolute coordinates
                        float_numbers = []
                        for num_idx, num in enumerate(numbers):
                            num = float(num)
                            if num_idx % 2 == 0:
                                # float_numbers.append(
                                #     float(num / smart_resize_width))
                                float_numbers.append(float(num / model_input_width))
                            else:
                                # float_numbers.append(
                                #     float(num / smart_resize_height))
                                float_numbers.append(float(num / model_input_height))

                        assert len(float_numbers) == 2, f"Argument {param_name}:{param} has wrong format."

                        action_inputs[param_name.strip()] = str(float_numbers)

                actions.append({
                    "reflection": reflection,
                    "thought": thought,
                    "action_type": action_type,
                    "action_inputs": action_inputs,
                    "generation": generation
                })

            return actions

        except Exception as e:
            print(f"Error in parsing UI-Tars generation:\n"
                  f"Generation: {generation}"
                  f"Error: {e}")

            return None

    @staticmethod
    def parse_function_call(function_call_str: str) -> Dict[str, Any] | None:
        """
        Given a function call string, e.g., "click(start_box='(100,100,200,200)')",
        parse it into a "function" and "args", e.g. {'function': 'click', 'args': {'start_box': '(100,100,200,200)'}}

        Returns:
            None if not parsable,
            Dictionary if parsable, consisting of:
            "function": str,
            "args": Dict[str, str]
        """
        # 1. Initialize Parser
        node = ast.parse(function_call_str, mode='eval')

        # 2. Ensure the string is actual expression
        if not isinstance(node, ast.Expression):
            return None

        call = node.body
        # Ensure it is specifically a FUNCTION CALL (e.g., "func()")
        # e.g., if the model output "x = 5" (Assignment), this would fail here.
        if not isinstance(call, ast.Call):
            return None

        # 3. Extract function name
        # Handles "click(...)" (Name) or "mouse.click(...)" (Attribute)
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            return None

        # 4. Extract arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg

            # Handle Python version differences for extracting values
            if isinstance(kw.value, ast.Constant):  # Python 3.8+
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # Old Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            "function": func_name,
            "args": kwargs,
        }

    @staticmethod
    def parse_point_coordinates(text: str) -> str:
        """
        Reference: ui_tars/action_parser.parse_point_coordinates
        Parses <point> tags from model output and converts them to (x,y) format.

        Example:
            Input:  "click(point='<point>500 300</point>') [EOS]"
            Output: "click(point='(500,300)')"
        """
        # 1. Remove the End-of-Sentence token
        text = text.replace("[EOS]", "")

        # 2. Remove <|box_start|>, <|box_end|> (legacy from previous instruction in osworld)
        text = text.replace("<|box_start|>", "").replace("<|box_end|>", "")

        # 3. Regex to match <point>X Y</point>
        # \s* allows for optional spaces around the numbers
        pattern = r"<point>\s*(\d+)\s+(\d+)\s*</point>"

        def format_match(match):
            x, y = match.groups()
            return f"({x},{y})"

        # 3. Replace and strip whitespace
        return re.sub(pattern, format_match, text).strip()

    @staticmethod
    def smart_resize(height: int,
                     width: int,
                     factor: int = 28,
                     min_pixels: int = 100 * 28 * 28,
                     max_pixels: int = 16384 * 28 * 28,
                     max_ratio: int = 200) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:
            1. Both dimensions (height and width) are divisible by 'factor'.
            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
            3. The aspect ratio of the image is maintained as closely as possible.
        """
        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
            )

        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    @staticmethod
    def convert_to_pyautogui(action_dict_list: List[Dict[str, Any]], image_width: int, image_height: int) -> str:
        """
        Reference: action_parser.parsing_response_to_pyautogui_code
        Convert each element in the result of parse_thought_and_action.
        Input:
            result: list of dictionaries, each consisting of "action_type" and "action_inputs" e.g.
            {
                "action_type": "hotkey",
                "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None,
            }
        Returns:
            String pyautogui command
        """
        pyautogui_code = f"import pyautogui\nimport time\n\ntime.sleep(1)\n"
        for action_dict_id, action_dict in enumerate(action_dict_list):
            action_type, action_inputs = action_dict.get("action_type"), action_dict.get("action_inputs", {})

            if action_type == "hotkey":
                hotkey = action_inputs.get("key", action_inputs.get("hotkey", ""))

                # process arrow keys
                if hotkey == "arrowleft":
                    hotkey = "left"
                elif hotkey == "arrowright":
                    hotkey = "right"
                elif hotkey == "arrowup":
                    hotkey = "up"
                elif hotkey == "arrowdown":
                    hotkey = "down"

                if hotkey:
                    keys = hotkey.split()  # e.g. "v ctrl" => ["v", "ctrl"]
                    keys = [k if k != "space" else " " for k in keys]
                    pyautogui_code += f"\npyautogui.hotkey({', '.join(repr(k) for k in keys)})"

            elif action_type in ["press", "keydown"]:
                if "key" in action_inputs:
                    key_to_press = action_inputs.get("key", "")
                else:
                    key_to_press = action_inputs.get("press", "")

                # process arrow keys
                if key_to_press == "arrowleft":
                    key_to_press = "left"
                elif key_to_press == "arrowright":
                    key_to_press = "right"
                elif key_to_press == "arrowup":
                    key_to_press = "up"
                elif key_to_press == "arrowdown":
                    key_to_press = "down"
                elif key_to_press == "space":
                    key_to_press = " "

                if key_to_press:
                    pyautogui_code += f"\npyautogui.keyDown({repr(key_to_press)})"

            elif action_type in ["release", "keyup"]:
                if "key" in action_inputs:
                    key_to_press = action_inputs.get("key", "")
                else:
                    key_to_press = action_inputs.get("press", "")

                # process arrow keys
                if key_to_press == "arrowleft":
                    key_to_press = "left"
                elif key_to_press == "arrowright":
                    key_to_press = "right"
                elif key_to_press == "arrowup":
                    key_to_press = "up"
                elif key_to_press == "arrowdown":
                    key_to_press = "down"
                elif key_to_press == "space":
                    key_to_press = " "

                if key_to_press:
                    pyautogui_code += f"\npyautogui.keyUp({repr(key_to_press)})"

            elif action_type == "type":
                content = action_inputs.get("content", "").strip()
                content = UITarsController.escape_single_quotes(content)
                stripped_content = content
                if content.endswith("\n") or content.endswith("\\n"):
                    stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
                if content:
                    # note original code had pyperclip + input_swap, we omit it here
                    pyautogui_code += f"\npyautogui.typewrite('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

            elif action_type in ["drag", "select"]:
                start_point = action_inputs.get("start_point", action_inputs.get("start_box"))  # e.g., '[0.595, 0.293]'
                end_point = action_inputs.get("end_point", action_inputs.get("end_box"))

                if start_point and end_point:
                    sx, sy = eval(start_point)
                    ex, ey = eval(end_point)

                    sx = round(float(sx) * image_width, 3)
                    sy = round(float(sy) * image_height, 3)
                    ex = round(float(ex) * image_width, 3)
                    ey = round(float(ey) * image_height, 3)
                    pyautogui_code += (
                        f"\npyautogui.moveTo({sx}, {sy})\n"
                        f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                    )

            elif action_type == "scroll":
                point = action_inputs.get("point", action_inputs.get("start_box"))  # e.g. '[0.595, 0.293]'
                if point:
                    x, y = eval(point)
                    x, y = round(float(x) * image_width, 3), round(float(y) * image_height, 3)
                else:
                    x, y = None, None

                direction = action_inputs.get("direction", "")
                if x is None or y is None:
                    if "up" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(5)"
                    elif "down" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(-5)"
                else:
                    if "up" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                    elif "down" in direction.lower():
                        pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

            elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
                point = action_inputs.get("point", action_inputs.get("start_box"))  # e.g. '[0.595, 0.293]'
                if point:
                    x, y = eval(point)
                    x, y = round(float(x) * image_width, 3), round(float(y) * image_height, 3)

                    if action_type == "left_single" or action_type == "click":
                        pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                    elif action_type == "left_double":
                        pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                    elif action_type == "right_single":
                        pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                    elif action_type == "hover":
                        pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

            elif action_type == "wait":
                pyautogui_code += f"\ntime.sleep(5)"

            elif action_type == "finished":
                pyautogui_code += f"\nDONE"

            else:
                pyautogui_code += f"\n# Unrecognized action type: {action_type}"

            return pyautogui_code

    @staticmethod
    def escape_single_quotes(text: str) -> str:
        # todo perhaps move this to util
        """
        Find all single quotes ' in the text, and if they are not escaped (i.e., no backlash \\ immediately before),
        escape them.
        """
        pattern = r"(?<!\\)'"
        return re.sub(pattern, r"\\'", text)

    @staticmethod
    def resize_image(image: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:
        # todo perhaps move this to util
        pixel_count = image.width * image.height
        if pixel_count > max_pixels:
            resize_factor = math.sqrt(max_pixels / pixel_count)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        elif pixel_count < min_pixels:
            resize_factor = math.sqrt(min_pixels / pixel_count)
            width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def generate_action(
            self,
            goal: str,
            current_screenshot: Union[bytes, Image.Image],
            history_images: List[Union[bytes, Image.Image]],
            history_responses: List[str],
        ) -> Dict[str, Any]:
        messages, resized_screenshot = self.prepare_generate_action_messages(
            goal, current_screenshot, history_images, history_responses
        )
        if isinstance(current_screenshot, bytes):
            current_screenshot = Image.open(BytesIO(current_screenshot))

        original_width, original_height = current_screenshot.size
        model_input_width, model_input_height = resized_screenshot.size

        num_generation, action_dict_list = 0, None
        while num_generation < self.max_retry_for_action_generation:
            responses, reasons = self.prompt_vlm_with_reason(messages)
            response, reason = responses[0], reasons[0]

            action_dict_list = self.parse_thought_and_action(response, model_input_width, model_input_height)
            if reason == "stop" and action_dict_list is not None:
                break

        if action_dict_list is None:
            # todo exception handling should be improved - maybe return empty result
            # todo all exception handling on ui-tars side should be done in generate_action as top-level
            # OpenAI problem or generation was unexpected
            raise ConnectionError("`generate_action` failed.")

        # todo convert action_inputs / action_type into pyautogui command
        pyautogui_code = self.convert_to_pyautogui(action_dict_list, original_width, original_height)

        return {
            "pyautogui_code": pyautogui_code,
            "action_dict_list": action_dict_list,
        }

    @staticmethod
    def prepare_generate_action_messages(
            goal: str,
            current_screenshot: Union[bytes, Image.Image],
            history_images: List[Union[bytes, Image.Image]],
            history_responses: List[str],
            history_n: int = 5,
            min_pixels: int = 100 * 28 * 28,
            max_pixels: int = 16384 * 28 * 28,
            max_context_window: int = 65536
    ) -> Tuple[List[Dict], Image.Image]:
        """
        Constructs the message list for UI-TARS 1.5-7B inference.

        Args:
            goal: The user's goal or command.
            current_screenshot: The latest screen observation.
            history_images: List of previous screenshots (excluding current).
            history_responses: List of previous model text responses.
            prompt_template: The UITARS prompt template (e.g., UITARS_USR_PROMPT_THOUGHT).
            action_space_str: The definition of available actions (e.g., UITARS_ACTION_SPACE).
            language: Language for the system prompt (default 'Chinese' per source).
            history_n: Max number of past images to include.
            min_pixels: Minimum pixel count for resizing (Qwen2-VL requirement).
            max_pixels: Maximum pixel count for resizing (Qwen2-VL requirement).
            max_context_window: Token limit approximation for dynamic image resizing.

        Returns: Tuple of
            List[Dict]: The formatted messages payload for the OpenAI/vLLM client.
            Image.Image: Resized image that is going to be given as input to the model
        """

        # 1. Format the Text Prompt
        # This corresponds to the user_prompt logic in the original predict method
        instruction_prompt = (
            # new
            f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform "
            f"the next action to complete the task.\n\n"

            f"## Output Format\n"
            f"```\n"
            f"Thought: ...\n"
            f"Action: ...\n"
            f"```\n\n"

            f"## Action Space\n\n"

            f"click(point='<point>x1 y1</point>')\n"
            f"left_double(point='<point>x1 y1</point>')\n"
            f"right_single(point='<point>x1 y1</point>')\n"
            f"drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')\n"
            f"hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in "
            f"one hotkey action.\n"
            f"type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse "
            f"the content in normal python string format. If you want to submit your input, use \\n at the end of "
            f"content.\n"
            f"scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information "
            f"on the `direction` side.\n"
            f"wait() #Sleep for 5s and take a screenshot to check for any changes.\n"
            f"finished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse "
            f"the content in normal python string format.\n\n\n"


            f"## Note\n"
            f"- Use English in `Thought` part.\n"
            f"- Write a small plan and finally summarize your next action (with its target element) in one sentence in "
            f"`Thought` part.\n\n"

            f"## User Instruction\n"
            f"{goal}\n"
        )

        # 2. Prepare Image List (History + Current)
        # We combine them to apply uniform resizing logic, then separate them for message construction
        full_image_list = history_images + [current_screenshot]

        # Apply history_n limit
        if len(full_image_list) > history_n:
            full_image_list = full_image_list[-history_n:]

        # 3. Dynamic Resolution Adjustment
        # Calculate max pixels based on context window to prevent OOM
        # Logic extracted from: max_image_nums_under_32k calculation in source
        processed_images: List[Image.Image] = []

        # Calculate dynamic max_pixels if history is long
        num_max_tokens_per_image = max_pixels / 28 * 28  # 1 token amounts to 28 * 28 pixels
        max_image_nums_under_limit = int(max_context_window * 0.75 / num_max_tokens_per_image)
        current_max_pixels = max_pixels

        if len(full_image_list) > max_image_nums_under_limit:
            # If too many images, reduce quality of all images to fit
            num_of_images = min(history_n, len(full_image_list))
            current_max_pixels = int(max_context_window * 0.75) * 28 * 28 // num_of_images

        # 4. Process Images (Load, Resize, Convert RGB)
        for img_obj in full_image_list:
            if isinstance(img_obj, bytes):
                image = Image.open(BytesIO(img_obj))
            else:
                image = img_obj

            # Resize logic
            image = process_image(image, min_pixels=min_pixels, max_pixels=current_max_pixels)

            processed_images.append(image)

        # 5. Construct Message Payload
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction_prompt}]
            }
        ]

        # 6. Interleave Images and History Responses
        if history_responses:
            # number of history images we can add: -1 to exclude the current image
            num_history_images_available = len(processed_images) - 1
            # where to start attaching image in history_responses
            attach_image_start_idx = len(history_responses) - num_history_images_available
            image_cursor = 0
            for history_idx, history_response in enumerate(history_responses):
                if history_idx >= attach_image_start_idx:
                    cur_image = processed_images[image_cursor]
                    encoded_string = image_to_base64(cur_image)

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"{encoded_string}"}}
                        ]
                    })
                    image_cursor += 1

                messages.append({
                    "role": "assistant",
                    "content": history_response
                })

        # 7. Add Current Observation (Final Image)
        final_image = processed_images[-1]
        encoded_final = image_to_base64(final_image)
        messages.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"{encoded_final}"}}]
        })

        return messages, final_image


if __name__ == "__main__":
    uitars_controller = UITarsController()

    # inputs
    goal = "Open the Chrome and navigate to nvidia.com."
    screenshot = Image.open("./annotated_images/startup.png")

    uitars_controller.generate_action(goal, screenshot, history_images=[], history_responses=[])

