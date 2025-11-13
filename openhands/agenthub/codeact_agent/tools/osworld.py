"""OSWorld tools for interacting with virtual machines.

This module provides tools for agents to control OSWorld virtual machines,
including mouse, keyboard, and screen operations through the OSWorld server API.
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import OSWORLD_TOOL_NAME

_OSWORLD_DESCRIPTION = """Interact with an OSWorld virtual machine through its server API. 
Use this tool to control the mouse, keyboard, take screenshots, or execute commands in a real VM environment.

The tool communicates with an OSWorld Flask server running inside a QEMU VM.
You can perform GUI automation tasks like clicking, typing, and taking screenshots.

Multiple actions can be chained, but they execute sequentially. Complex sequences (>3 actions) 
may fail or produce unexpected results.
"""

_OSWORLD_TOOL_DESCRIPTION = """
OSWorld VM Control Functions

The OSWorld server exposes actions through HTTP endpoints. Each action is specified as a dictionary
with 'action_type' and 'parameters' fields.

# Mouse Control Actions

CLICK Action:
    Description: Click the mouse at a specific position
    Parameters:
        x (int): X coordinate on screen
        y (int): Y coordinate on screen  
        button (str): Mouse button - 'left', 'right', or 'middle' (default: 'left')
        clicks (int): Number of clicks (default: 1)
    Example:
        {"action_type": "CLICK", "parameters": {"x": 500, "y": 300, "button": "left"}}

DOUBLE_CLICK Action:
    Description: Double-click at a position
    Parameters:
        x (int): X coordinate
        y (int): Y coordinate
        button (str): Mouse button (default: 'left')
    Example:
        {"action_type": "DOUBLE_CLICK", "parameters": {"x": 500, "y": 300}}

RIGHT_CLICK Action:
    Description: Right-click at a position
    Parameters:
        x (int): X coordinate
        y (int): Y coordinate
    Example:
        {"action_type": "RIGHT_CLICK", "parameters": {"x": 500, "y": 300}}

DRAG Action:
    Description: Drag from one position to another
    Parameters:
        start_x (int): Starting X coordinate
        start_y (int): Starting Y coordinate
        end_x (int): Ending X coordinate
        end_y (int): Ending Y coordinate
        button (str): Mouse button to hold (default: 'left')
        duration (float): Time to complete drag in seconds (default: 0.5)
    Example:
        {"action_type": "DRAG", "parameters": {"start_x": 100, "start_y": 100, "end_x": 500, "end_y": 500}}

MOVE_TO Action:
    Description: Move mouse to a position without clicking
    Parameters:
        x (int): X coordinate
        y (int): Y coordinate
        duration (float): Time to move in seconds (default: 0.0)
    Example:
        {"action_type": "MOVE_TO", "parameters": {"x": 500, "y": 300}}

SCROLL Action:
    Description: Scroll the mouse wheel
    Parameters:
        clicks (int): Amount to scroll (positive = up, negative = down)
        x (int, optional): X position to scroll at
        y (int, optional): Y position to scroll at
    Example:
        {"action_type": "SCROLL", "parameters": {"clicks": -3}}

# Keyboard Control Actions

TYPING Action:
    Description: Type a string of text
    Parameters:
        text (str): Text to type
        interval (float): Delay between keypresses in seconds (default: 0.0)
    Example:
        {"action_type": "TYPING", "parameters": {"text": "Hello World"}}

PRESS Action:
    Description: Press a keyboard key or key combination
    Parameters:
        key (str or list): Key name or list of keys to press simultaneously
        presses (int): Number of times to press (default: 1)
        interval (float): Delay between presses (default: 0.0)
    Valid key names: 'enter', 'esc', 'tab', 'space', 'backspace', 'delete', 'up', 'down',
    'left', 'right', 'home', 'end', 'pageup', 'pagedown', 'f1'-'f12', 'shift', 'ctrl', 'alt',
    'win', 'command', 'a'-'z', '0'-'9'
    Example:
        {"action_type": "PRESS", "parameters": {"key": "enter"}}
        {"action_type": "PRESS", "parameters": {"key": ["ctrl", "c"]}}

HOTKEY Action:
    Description: Press a keyboard shortcut (like Ctrl+C)
    Parameters:
        keys (list): List of keys to press together
    Example:
        {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "alt", "delete"]}}

# Screen Operations

GET_SCREENSHOT Action:
    Description: Take a screenshot of the VM screen
    Parameters: None
    Returns: PNG image data
    Example:
        {"action_type": "GET_SCREENSHOT", "parameters": {}}

GET_SCREEN_SIZE Action:
    Description: Get the screen dimensions
    Parameters: None
    Returns: {"width": int, "height": int}
    Example:
        {"action_type": "GET_SCREEN_SIZE", "parameters": {}}

GET_ACCESSIBILITY_TREE Action:
    Description: Get the UI accessibility tree (XML structure of UI elements)
    Parameters: None
    Returns: XML string representing UI elements
    Example:
        {"action_type": "GET_ACCESSIBILITY_TREE", "parameters": {}}

# Command Execution

EXECUTE_PYTHON Action:
    Description: Execute Python code inside the VM
    Parameters:
        code (str): Python code to execute
    Example:
        {"action_type": "EXECUTE_PYTHON", "parameters": {"code": "import os; print(os.listdir('/'))"}}

EXECUTE_BASH Action:
    Description: Execute a bash command inside the VM
    Parameters:
        command (str): Bash command to execute
        timeout (int): Timeout in seconds (default: 30)
    Example:
        {"action_type": "EXECUTE_BASH", "parameters": {"command": "ls -la /home"}}

# File Operations

GET_FILE Action:
    Description: Download a file from the VM
    Parameters:
        path (str): Path to file in VM
    Returns: File content as bytes
    Example:
        {"action_type": "GET_FILE", "parameters": {"path": "/home/user/document.txt"}}

UPLOAD_FILE Action:
    Description: Upload a file to the VM
    Parameters:
        path (str): Destination path in VM
        content (bytes or str): File content
    Example:
        {"action_type": "UPLOAD_FILE", "parameters": {"path": "/home/user/file.txt", "content": "Hello"}}

# VM Control

GET_VM_PLATFORM Action:
    Description: Get the VM's operating system platform
    Parameters: None
    Returns: "Linux", "Windows", or "Darwin"
    Example:
        {"action_type": "GET_VM_PLATFORM", "parameters": {}}

START_RECORDING Action:
    Description: Start recording the VM screen
    Parameters: None
    Example:
        {"action_type": "START_RECORDING", "parameters": {}}

STOP_RECORDING Action:
    Description: Stop recording and save video
    Parameters:
        output_path (str): Path to save video file
    Example:
        {"action_type": "STOP_RECORDING", "parameters": {"output_path": "/tmp/recording.mp4"}}

# Usage Notes

1. Actions are sent to the OSWorld Flask server running inside the VM
2. The server uses pyautogui for mouse/keyboard control
3. All coordinates are in screen pixels (0,0 is top-left)
4. Multiple actions should be sent separately for better error handling
5. Screenshots can be large - use sparingly
6. The VM must have the OSWorld server running on port 5000
"""

OSWorldTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=OSWORLD_TOOL_NAME,
        description=_OSWORLD_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'object',
                    'description': (
                        'The action to execute in the OSWorld VM. Must be a dictionary with '
                        '"action_type" and "parameters" fields.\n'
                        + _OSWORLD_TOOL_DESCRIPTION
                    ),
                    'properties': {
                        'action_type': {
                            'type': 'string',
                            'description': 'Type of action to perform',
                            'enum': [
                                'CLICK', 'DOUBLE_CLICK', 'RIGHT_CLICK', 'DRAG', 'MOVE_TO', 'SCROLL',
                                'TYPING', 'PRESS', 'HOTKEY',
                                'GET_SCREENSHOT', 'GET_SCREEN_SIZE', 'GET_ACCESSIBILITY_TREE',
                                'EXECUTE_PYTHON', 'EXECUTE_BASH',
                                'GET_FILE', 'UPLOAD_FILE',
                                'GET_VM_PLATFORM', 'START_RECORDING', 'STOP_RECORDING'
                            ]
                        },
                        'parameters': {
                            'type': 'object',
                            'description': 'Parameters for the action'
                        }
                    },
                    'required': ['action_type', 'parameters']
                }
            },
            'required': ['action'],
        },
    ),
)

