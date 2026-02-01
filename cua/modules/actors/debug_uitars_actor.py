import re
from argparse import Namespace
from io import BytesIO
from typing import List, Tuple, Union, Dict, Any, Optional

import ipdb
from PIL import Image
from openai import OpenAI

from cua.modules.util import process_image, image_to_base64
from cua.modules.actors.util_uitars import escape_single_quotes, parse_point_coordinates, parse_function_call
from openhands.core.logger import openhands_logger


logger = openhands_logger.getChild('uitars_actor')


class UITarsActor:
    """
    Interface for UI-Tars model
    References
    - https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/uitars_agent.py
    - https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/action_parser.py
    """

    def __init__(self, args: Namespace):
        self.client = OpenAI(
            base_url=f"http://{args.actor_node}:8000/v1",
            api_key="gen",
        )
        self.model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
        self.max_retry_for_action_generation = args.max_retry_for_action_generation

    def prompt_vlm_with_reason(
            self, messages: List, n: int = 1, temperature: float = 0.3, max_completion_tokens: int = 4096,
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
            top_p=0.9,
            max_completion_tokens=max_completion_tokens,
            timeout=int(600),
        )

        responses = [choice.message.content for choice in chat_response.choices]
        reasons = [choice.finish_reason for choice in chat_response.choices]

        return responses, reasons

    def generate_action(
            self,
            goal: str,
            current_screenshot: Union[bytes, Image.Image],
            history_images: List[Union[bytes, Image.Image]],
            history_responses: List[str],
        ) -> Optional[Dict[str, Any]]:
        """
        Returns: a dictionary of
            "pyautogui_command": a string command to send to EnvController
            "action_generation": {
                "generation": str (raw generation),
                "reflection": str,
                "thought": str,
                "parsed_actions": list of {
                    "action_type": str (e.g., "click"),
                    "action_inputs": dict (e.g., {"start_box": "[0.089, 0.424]"}),
                },
            }
        """
        messages, resized_screenshot = self.prepare_generate_action_messages(
            goal, current_screenshot, history_images, history_responses
        )
        if isinstance(current_screenshot, bytes):
            current_screenshot = Image.open(BytesIO(current_screenshot))

        original_width, original_height = current_screenshot.size
        model_input_width, model_input_height = resized_screenshot.size

        num_generation, action_generation = 0, None
        while num_generation < self.max_retry_for_action_generation:
            try:
                responses, reasons = self.prompt_vlm_with_reason(messages, n=1, temperature=0.3)
                response, reason = responses[0], reasons[0]

                action_generation = self.parse_action_generation(response, model_input_width, model_input_height)
                if (
                    reason == "stop" and
                    action_generation is not None and
                    all(self.validate_parsed_action(pa["action_type"], pa["action_inputs"])
                        for pa in action_generation["parsed_actions"])
                ):
                    break

                # todo remove this when not debugging
                # the generation failed to meet the required syntax
                ipdb.set_trace()
                pass

            except Exception as e:
                logger.warning(f"Error in generate_action: {e}")

                # todo remove below when not debugging
                ipdb.set_trace()
                pass

            finally:
                num_generation += 1

        if action_generation is None:
            # OpenAI problem or generation was unexpected
            logger.warning("`generate_action` failed.")
            return None

        pyautogui_code = self.convert_to_pyautogui(action_generation["parsed_actions"], original_width, original_height)

        return {
            "pyautogui_command": pyautogui_code,
            "action_generation": action_generation,
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
            history_n: Max number of past images to include.
            min_pixels: Minimum pixel count for resizing (Qwen2-VL requirement).
            max_pixels: Maximum pixel count for resizing (Qwen2-VL requirement).
            max_context_window: Token limit approximation for dynamic image resizing.

        Returns: Tuple of
            List[Dict]: The formatted messages payload for the OpenAI/vLLM client.
            Image.Image: Resized image that is going to be given as input to the model
        """

        # 1. Format the Text Prompt
        instruction_prompt = (
            # older version used for OSWorld
            f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform "
            f"the next action to complete the task.\n\n"

            f"## Output Format\n"
            f"```\n"
            f"Thought: ...\n"
            f"Action: ...\n"
            f"```\n\n"

            f"## Action Space\n"
            f"click(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
            f"left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
            f"right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')\n"
            f"drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')\n"
            f"hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in "
            f"one hotkey action.\n"
            f"type(content='') #If you want to submit your input, use \\n at the end of `content`.\n"
            f"scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')"
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

        # separate history images and current screenshot
        processed_history_images, processed_current_screenshot = processed_images[:-1], processed_images[-1]

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
            image_cursor = 0
            for history_idx, history_response in enumerate(history_responses):
                if history_idx >= len(history_responses) - len(processed_history_images):
                    cur_image = processed_history_images[image_cursor]
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

        # 7. Add current screenshot
        encoded_final = image_to_base64(processed_current_screenshot)
        messages.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"{encoded_final}"}}]
        })

        return messages, processed_current_screenshot

    @staticmethod
    def parse_action_generation(generation, model_input_width: int, model_input_height: int) -> Union[None, Dict[str, Any]]:
        """
        Parse UI-Tars action generation.
        Reference: uitars_agent.parse_action_qwen2vl
        Inputs:
            generation: raw generation generated by UI-TARS
            model_input_width: actual width of the last image given as input to UI-TARS
            model_input_height: actual height of the last image given as input to UI-TARS
        Returns:
            None if error in parsing
            Dictionary if successful, {
                "generation": str (raw generation),
                "reflection": str,
                "thought": str,
                "parsed_actions": list of {
                    "action_type": str (e.g., "click"),
                    "action_inputs": dict (e.g., {"start_box": "[0.089, 0.424]"}),
                },
            }
        """
        try:
            assert "Action:" in generation, f"`Action:` does not exist in generation {generation}"

            # 1. Parse thought, reflection (optional), action from generation
            generation = parse_point_coordinates(generation)

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
                    action_str = escape_single_quotes(content)
                    action_str = f"type(content='{action_str}')"

                if not action_str.strip().endswith(")"):
                    action_str = action_str.strip() + ")"

                all_action.append(action_str)

            # 3. Parse actions in all_action
            parsed_actions = [parse_function_call(action.replace("\n", "\\n").lstrip())
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

                    # all coordinates must go through the below process, for coordinate transformation
                    # e.g. {"point": "(500,300)"} => {"point": "[0.234, 0.313]"}
                    # e.g. {"start_box": "(500,300)"} => {"start_box": "[0.234, 0.313]"}
                    if "point" in param_name or "box" in param_name:
                        bbox_str = param  # e.g. "(595,25)"

                        # Remove parentheses and split the string by commas
                        numbers = bbox_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")

                        # UI-TARS-1.5 outputs absolute coordinates specific to the given image
                        float_numbers = []
                        for num_idx, num in enumerate(numbers):
                            num = float(num)
                            if num_idx % 2 == 0:
                                float_numbers.append(float(num / model_input_width))
                            else:
                                float_numbers.append(float(num / model_input_height))

                        assert len(float_numbers) == 2, f"Argument {param_name}:{param} has wrong format."

                        action_inputs[param_name.strip()] = str(float_numbers)

                actions.append({
                    "action_type": action_type,
                    "action_inputs": action_inputs,
                })

            return {
                "generation": generation,
                "reflection": reflection,
                "thought": thought,
                "parsed_actions": actions,
            }

        except Exception as e:
            print(f"Error in parsing UI-Tars generation:\n"
                  f"Generation: {generation}\n"
                  f"Error: {e}")

            return None

    @staticmethod
    def validate_parsed_action(action_type: str, action_inputs: Dict[str, Any]) -> bool:
        """
        Helper function to validate whether the output of parse_action_generation, i.e.,
        each element of action_generation["parsed_actions"] qualifies as input to convert_to_pyautogui
        """
        if action_type == "hotkey":
            if "key" not in action_inputs and "hotkey" not in action_inputs:
                logger.debug(f"`key` or `hotkey` not in action_inputs for action_type hotkey")
                return False

        elif action_type in ["press", "keydown", "release", "keyup"]:
            if "key" not in action_inputs and "press" not in action_inputs:
                logger.debug(f"`key` or `press` not in action_inputs for action_type {action_type}")
                return False

        elif action_type == "type":
            if "content" not in action_inputs:
                logger.debug(f"`content` not in action_inputs for action_type type")
                return False

        elif action_type in ["drag", "select"]:
            if "start_point" not in action_inputs and "start_box" not in action_inputs:
                logger.debug(f"`start_point` or `start_box` not in action_inputs for action_type {action_type}")
                return False

            if "end_point" not in action_inputs and "end_box" not in action_inputs:
                logger.debug(f"`end_point` or `end_box` not in action_inputs for action_type {action_type}")
                return False

            start_point = eval(action_inputs.get("start_point", action_inputs.get("start_box")))
            if (not isinstance(start_point, tuple) and not isinstance(start_point, list)) or len(start_point) != 2:
                logger.debug(f"action_type: {action_type} / action_inputs: {action_inputs}: {start_point} unexpected")
                return False

            end_point = eval(action_inputs.get("end_point", action_inputs.get("end_box")))
            if (not isinstance(end_point, tuple) and not isinstance(end_point, list)) or len(end_point) != 2:
                logger.debug(f"action_type: {action_type} / action_inputs: {action_inputs}: {end_point} unexpected")
                return False

        elif action_type == "scroll":
            if "point" not in action_inputs and "start_box" not in action_inputs:
                logger.debug(f"`point` or `start_box` not in action_inputs for action_type scroll")
                return False

            if "direction" not in action_inputs:
                logger.debug(f"`direction` not in action_inputs for action_type scroll")
                return False

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            if "point" not in action_inputs and "start_box" not in action_inputs:
                logger.debug(f"`point` or `start_box` not in action_inputs for action_type {action_type}")
                return False

            point = eval(action_inputs.get("point", action_inputs.get("start_box")))
            if (not isinstance(point, tuple) and not isinstance(point, list)) or len(point) != 2:
                logger.debug(f"action_type: {action_type} / action_inputs: {action_inputs}: {point} unexpected")
                return False

        elif action_type in ["wait", "finished"]:
            pass

        else:
            logger.debug(f"Unrecognized action_type: {action_type}")
            return False

        return True

    @staticmethod
    def convert_to_pyautogui(parsed_actions: List[Dict[str, Any]], image_width: int, image_height: int) -> str:
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
                },
            }
        Returns:
            String pyautogui command
        """
        pyautogui_code = f"import pyautogui\nimport time\n\ntime.sleep(1)\n"
        for action_dict_id, action_dict in enumerate(parsed_actions):
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
                content = escape_single_quotes(content)
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

                    sx, sy = round(float(sx) * image_width, 3), round(float(sy) * image_height, 3)
                    ex, ey = round(float(ex) * image_width, 3), round(float(ey) * image_height, 3)

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
                point = action_inputs.get("point", action_inputs.get("start_box", ""))  # e.g. '[0.595, 0.293]'

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
                pyautogui_code += f"\n# DONE"

            else:
                pyautogui_code += f"\n# Unrecognized action type: {action_type}"

            return pyautogui_code
