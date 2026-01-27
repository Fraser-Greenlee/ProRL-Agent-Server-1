import ast
import re
from typing import Dict, Any


def escape_single_quotes(text: str) -> str:
    """
    Find all single quotes ' in the text, and if they are not escaped (i.e., no backlash \\ immediately before),
    escape them.
    """
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def parse_point_coordinates(text: str) -> str:
    """
    Reference: ui_tars/action_parser.parse_point_coordinates

    (1) Removes [EOS].
    (2) Remove <point>, <|box_start|>, <|box_end|> tags, and converts the coordinates inside into (x,y) format.

    Example:
        Input:  "click(point='<point>500 300</point>') [EOS]"
        Output: "click(point='(500,300)')"
    """
    # 1. Remove the End-of-Sentence token
    text = text.replace("[EOS]", "")

    # 2. Replace <|box_start|>, <|box_end|> (legacy from previous instruction in osworld)
    text = text.replace("<|box_start|>", "<point>").replace("<|box_end|>", "</point>")

    # 3. Regex to match <point>X Y</point>
    # \s* allows for optional spaces around the numbers
    pattern = r"<point>\s*(\d+)\s+(\d+)\s*</point>"

    def format_match(match):
        x, y = match.groups()
        return f"({x},{y})"

    # 3. Replace and strip whitespace
    return re.sub(pattern, format_match, text).strip()


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

