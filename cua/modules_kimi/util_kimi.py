"""
Kimi-K2.5 response parsing utilities.
Copied from the reference KimiAgent:
  https://github.com/jayl940712/OSWorld/blob/fix/mm_agents/kimi/kimi_agent.py
"""
import ast
import re
import traceback
from typing import Dict, List, Tuple

from openhands.core.logger import openhands_logger

logger = openhands_logger.getChild('util_kimi')


def parse_response_to_cot_and_action(
    response: Dict[str, str],
    screen_size: Tuple[int, int],
    coordinate_type: str,
    thinking: bool,
) -> Tuple[str, List[str], Dict]:
    """
    Parse response including Observation, Thought, Action and code block.

    Args:
        response: dict with "content" and "reasoning_content" keys.
        screen_size: (width, height) of the VM screen.
        coordinate_type: "relative" or "absolute".
        thinking: whether reasoning/thinking mode is enabled.

    Returns:
        Tuple of (low_level_instruction, pyautogui_actions, sections) where:
            low_level_instruction: action description string or error string
            pyautogui_actions: list of code strings, or ["WAIT"], ["DONE"], ["FAIL"]
            sections: dict with "thought", "action", "code", "original_code" keys
    """
    input_string = (response.get('content') or '').lstrip()

    # vLLM sometimes leaks thinking content into the content field with </think> tag.
    # Strip everything up to and including </think> so ## Action regex can match.
    think_end = input_string.rfind('</think>')
    if think_end != -1:
        input_string = input_string[think_end + len('</think>'):].lstrip()

    sections = {}
    try:
        if thinking:
            if 'reasoning_content' in response:
                thought = response['reasoning_content'].strip()
            else:
                thought = (response.get('reasoning') or '').strip()
            sections['thought'] = thought
            # Remove extra content before ## Action
            m = re.search(r"^##\s*Action\b", input_string, flags=re.MULTILINE)
            if m:
                input_string = input_string[m.start():]
        else:
            thought = re.search(
                r'^##\s*Thought\s*:?[\n\r]+(.*?)(?=^##\s*Action:|^##|\Z)',
                input_string, re.DOTALL | re.MULTILINE
            )
            if thought:
                sections['thought'] = thought.group(1).strip()
            else:
                sections['thought'] = ""

        action_match = re.search(
            r'^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)',
            input_string, re.DOTALL | re.MULTILINE
        )
        if action_match:
            action = action_match.group(1).strip()
            sections['action'] = action.strip()

        code_blocks = re.findall(
            r'```(?:code|python)?\s*(.*?)\s*```', input_string, re.DOTALL | re.IGNORECASE
        )
        if not code_blocks:
            logger.error("No code blocks found in the input string")
            return (
                f"<Error>: no code blocks found in the input string: {input_string}",
                ["FAIL"],
                sections,
            )

        code_block = code_blocks[-1].strip()
        sections['original_code'] = code_block

        if "computer.wait" in code_block.lower():
            sections["code"] = "WAIT"
            return sections.get('action', ''), ["WAIT"], sections
        elif "computer.terminate" in code_block.lower():
            lower_block = code_block.lower()
            if ("failure" in lower_block) or ("fail" in lower_block):
                sections['code'] = "FAIL"
                return code_block, ["FAIL"], sections
            elif "success" in lower_block:
                sections['code'] = "DONE"
                return code_block, ["DONE"], sections
            else:
                logger.error(
                    "Terminate action found but no specific status provided in code block"
                )
                return (
                    f"<Error>: terminate action found but no specific status: {input_string}",
                    ["FAIL"],
                    sections,
                )

        corrected_code = code_block
        sections['code'] = project_coordinate_to_absolute_scale(
            corrected_code,
            screen_width=screen_size[0],
            screen_height=screen_size[1],
            coordinate_type=coordinate_type,
        )

        if 'code' not in sections or not sections['code']:
            logger.error("Missing required code section")
            return f"<Error>: no code parsed: {input_string}", ["FAIL"], sections

        if 'action' not in sections or not sections['action']:
            sections['action'] = ""

        return sections['action'], [sections['code']], sections

    except Exception as e:
        error_message = (
            f"<Error>: parsing response: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            f"Input string: {input_string}"
        )
        logger.exception(error_message)
        return error_message, ['FAIL'], sections


def project_coordinate_to_absolute_scale(
    pyautogui_code_relative_coordinates: str,
    screen_width: int,
    screen_height: int,
    coordinate_type: str = "relative",
) -> str:
    """
    Convert the relative coordinates in the pyautogui code to absolute coordinates
    based on the logical screen size.
    """
    def _coordinate_projection(x, y, screen_width, screen_height, coordinate_type):
        if x <= 1.0 and y <= 1.0:
            return int(round(x * screen_width)), int(round(y * screen_height))
        else:
            return int(round(x)), int(round(y))

    pattern = r'(pyautogui\.\w+\([^\)]*\))'
    matches = re.findall(pattern, pyautogui_code_relative_coordinates)

    new_code = pyautogui_code_relative_coordinates

    for full_call in matches:
        func_name_pattern = r'(pyautogui\.\w+)\((.*)\)'
        func_match = re.match(func_name_pattern, full_call, re.DOTALL)
        if not func_match:
            continue

        func_name = func_match.group(1)
        args_str = func_match.group(2)

        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            parsed_args = parsed.args
            parsed_keywords = parsed.keywords
        except SyntaxError:
            return pyautogui_code_relative_coordinates

        function_parameters = {
            'click': ['x', 'y', 'clicks', 'interval', 'button', 'duration', 'pause'],
            'rightClick': ['x', 'y', 'duration', 'tween', 'pause'],
            'middleClick': ['x', 'y', 'duration', 'tween', 'pause'],
            'doubleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
            'tripleClick': ['x', 'y', 'interval', 'button', 'duration', 'pause'],
            'moveTo': ['x', 'y', 'duration', 'tween', 'pause'],
            'dragTo': ['x', 'y', 'duration', 'button', 'mouseDownUp', 'pause'],
        }

        func_base_name = func_name.split('.')[-1]
        param_names = function_parameters.get(func_base_name, [])

        args = {}
        for idx, arg in enumerate(parsed_args):
            if idx < len(param_names):
                param_name = param_names[idx]
                arg_value = ast.literal_eval(arg)
                args[param_name] = arg_value

        try:
            for kw in parsed_keywords:
                param_name = kw.arg
                arg_value = ast.literal_eval(kw.value)
                args[param_name] = arg_value
        except Exception as e:
            logger.error(f"Error parsing keyword arguments: {e}")
            return pyautogui_code_relative_coordinates

        updated = False
        if 'x' in args and 'y' in args:
            try:
                x_rel = float(args['x'])
                y_rel = float(args['y'])
                x_abs, y_abs = _coordinate_projection(
                    x_rel, y_rel, screen_width, screen_height, coordinate_type
                )
                args['x'] = x_abs
                args['y'] = y_abs
                updated = True
            except ValueError:
                pass

        if updated:
            reconstructed_args = []
            for idx, param_name in enumerate(param_names):
                if param_name in args:
                    arg_value = args[param_name]
                    if isinstance(arg_value, str):
                        arg_repr = f"'{arg_value}'"
                    else:
                        arg_repr = str(arg_value)
                    reconstructed_args.append(arg_repr)
                else:
                    break

            used_params = set(param_names[:len(reconstructed_args)])
            for kw in parsed_keywords:
                if kw.arg not in used_params:
                    arg_value = args[kw.arg]
                    if isinstance(arg_value, str):
                        arg_repr = f"{kw.arg}='{arg_value}'"
                    else:
                        arg_repr = f"{kw.arg}={arg_value}"
                    reconstructed_args.append(arg_repr)

            new_args_str = ', '.join(reconstructed_args)
            new_full_call = f"{func_name}({new_args_str})"
            new_code = new_code.replace(full_call, new_full_call)

    return new_code
