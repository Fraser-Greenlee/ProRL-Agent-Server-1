"""
OSWorld Tools

Provides comprehensive tools for interacting with OSWorld virtual machines.
These tools expose all PythonController methods from the OSWorld framework,
including mouse/keyboard control, file operations, screenshots, and more.

Based on: osworld/desktop_env/controllers/python.py
"""

from openhands.llm.tool_names import OSWORLD_TOOL_NAME

OSWORLD_TOOLS_DOCSTRING = f'''
# {OSWORLD_TOOL_NAME}

The `{OSWORLD_TOOL_NAME}` tool allows you to interact with a virtual desktop environment (Ubuntu or Windows)
running inside a QEMU VM. This tool provides comprehensive control over the desktop including:

## Available Methods

### Mouse & Keyboard Actions

1. **execute_action** - Execute PyAutoGUI actions for mouse and keyboard control
   - Actions: CLICK, DOUBLE_CLICK, RIGHT_CLICK, MOVE_TO, DRAG_TO, SCROLL
   - Keyboard: TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY
   - Parameters: action dict with action_type and parameters

   Example:
   ```python
   osworld(
       method="execute_action",
       params={{
           "action": {{
               "action_type": "CLICK",
               "parameters": {{"x": 500, "y": 300, "button": "left"}}
           }}
       }}
   )
   ```

   ```python
   osworld(
       method="execute_action",
       params={{
           "action": {{
               "action_type": "TYPING",
               "parameters": {{"text": "Hello World"}}
           }}
       }}
   )
   ```

   ```python
   osworld(
       method="execute_action",
       params={{
           "action": {{
               "action_type": "HOTKEY",
               "parameters": {{"keys": ["ctrl", "c"]}}
           }}
       }}
   )
   ```

### Screen & Observation

2. **get_screenshot** - Capture a screenshot of the VM desktop (with cursor)
   - Returns: Screenshot image as base64

   Example:
   ```python
   osworld(method="get_screenshot", params={{}})
   ```

3. **get_accessibility_tree** - Get the accessibility tree of UI elements
   - Returns: Accessibility tree structure

   Example:
   ```python
   osworld(method="get_accessibility_tree", params={{}})
   ```

4. **get_terminal_output** - Get current terminal output from the VM
   - Returns: Terminal text output

   Example:
   ```python
   osworld(method="get_terminal_output", params={{}})
   ```

### File Operations

5. **get_file** - Download a file from the VM
   - Parameters:
     - file_path (str): Path to the file in the VM
   - Returns: File content as base64

   Example:
   ```python
   osworld(
       method="get_file",
       params={{"file_path": "/home/user/document.txt"}}
   )
   ```

### Script Execution

6. **execute_python_command** - Execute a raw Python command in the VM
   - Parameters:
     - command (str): Python command to execute
   - Returns: Command output

   Example:
   ```python
   osworld(
       method="execute_python_command",
       params={{"command": "print('Hello from VM')"}}
   )
   ```

7. **run_python_script** - Execute a Python script in the VM
   - Parameters:
     - script (str): Python script content
   - Returns: Script output and errors

   Example:
   ```python
   osworld(
       method="run_python_script",
       params={{
           "script": """
import os
print(os.getcwd())
print(os.listdir('.'))
"""
       }}
   )
   ```

8. **run_bash_script** - Execute a bash script in the VM
   - Parameters:
     - script (str): Bash script content
     - timeout (int, optional): Execution timeout in seconds (default: 30)
     - working_dir (str, optional): Working directory for execution
   - Returns: Script output, errors, and return code

   Example:
   ```python
   osworld(
       method="run_bash_script",
       params={{
           "script": "ls -la\\necho 'Done'",
           "timeout": 60,
           "working_dir": "/home/user"
       }}
   )
   ```

### Screen Recording

9. **start_recording** - Start recording the VM screen
   - Returns: Success/failure status

   Example:
   ```python
   osworld(method="start_recording", params={{}})
   ```

10. **end_recording** - Stop recording and get the video
    - Parameters:
      - dest (str, optional): Destination path for the recording
    - Returns: Video file content

    Example:
    ```python
    osworld(
        method="end_recording",
        params={{"dest": "/tmp/recording.mp4"}}
    )
    ```

### VM Information

11. **get_vm_platform** - Get the VM's operating system platform
    - Returns: Platform name (e.g., "Linux", "Windows")

    Example:
    ```python
    osworld(method="get_vm_platform", params={{}})
    ```

12. **get_vm_screen_size** - Get the VM screen dimensions
    - Returns: Width and height of the screen

    Example:
    ```python
    osworld(method="get_vm_screen_size", params={{}})
    ```

13. **get_vm_window_size** - Get a specific application window size
    - Parameters:
      - app_class_name (str): Application class name
    - Returns: Window width and height

    Example:
    ```python
    osworld(
        method="get_vm_window_size",
        params={{"app_class_name": "firefox"}}
    )
    ```

14. **get_vm_wallpaper** - Get the VM's desktop wallpaper image
    - Returns: Wallpaper image as base64

    Example:
    ```python
    osworld(method="get_vm_wallpaper", params={{}})
    ```

15. **get_vm_desktop_path** - Get the path to the desktop directory
    - Returns: Desktop directory path

    Example:
    ```python
    osworld(method="get_vm_desktop_path", params={{}})
    ```

16. **get_vm_directory_tree** - List directory contents in the VM
    - Parameters:
      - path (str): Directory path to list
    - Returns: Directory tree structure

    Example:
    ```python
    osworld(
        method="get_vm_directory_tree",
        params={{"path": "/home/user/Documents"}}
    )
    ```

## Action Types for execute_action

When using `execute_action`, specify one of these action_type values:

### Mouse Actions
- **CLICK**: Click mouse button
  - Parameters: x, y, button ("left"/"right"/"middle"), num_clicks (optional)
- **DOUBLE_CLICK**: Double-click at position
  - Parameters: x, y
- **RIGHT_CLICK**: Right-click at position
  - Parameters: x, y
- **MOVE_TO**: Move mouse to position
  - Parameters: x, y
- **DRAG_TO**: Drag mouse to position
  - Parameters: x, y
- **MOUSE_DOWN**: Press mouse button down
  - Parameters: button ("left"/"right"/"middle")
- **MOUSE_UP**: Release mouse button
  - Parameters: button ("left"/"right"/"middle")
- **SCROLL**: Scroll horizontally and/or vertically
  - Parameters: dx (horizontal), dy (vertical)

### Keyboard Actions
- **TYPING**: Type text string
  - Parameters: text (string to type)
- **PRESS**: Press and release a key
  - Parameters: key (key name like "enter", "tab", "a")
- **KEY_DOWN**: Press a key down
  - Parameters: key
- **KEY_UP**: Release a key
  - Parameters: key
- **HOTKEY**: Press key combination
  - Parameters: keys (list of keys like ["ctrl", "c"])

## Common Keyboard Keys

Supported keys include:
- Letters: "a", "b", "c", ..., "z"
- Numbers: "0", "1", ..., "9"
- Function keys: "f1", "f2", ..., "f12"
- Modifiers: "ctrl", "alt", "shift", "win" (or "command" on Mac)
- Special: "enter", "tab", "space", "backspace", "delete", "esc"
- Navigation: "up", "down", "left", "right", "home", "end", "pageup", "pagedown"

## Usage Guidelines

1. **Start with observation**: Use `get_screenshot` to see the current state
2. **Locate elements**: Use `get_accessibility_tree` to find UI elements
3. **Interact**: Use `execute_action` to click, type, etc.
4. **Verify**: Check results with another screenshot or terminal output
5. **File operations**: Use bash/python scripts for complex file tasks
6. **Recording**: Start recording before demos, stop when done

## Error Handling

All methods return observations with:
- Success: content field contains the result
- Failure: error message in content field, exit_code > 0

Always check the observation to verify your action succeeded before proceeding.
'''

OSWORLD_TOOLS_DESCRIPTION = f'''Use the `{OSWORLD_TOOL_NAME}` tool to interact with a virtual desktop environment running in a QEMU VM.

This tool supports:
- Mouse and keyboard control (click, type, hotkeys)
- Screen capture and accessibility tree
- File operations and script execution
- Screen recording
- VM information queries

Call pattern: `{OSWORLD_TOOL_NAME}(method="<method_name>", params={{"key": "value"}})`

See the docstring for all available methods and detailed examples.
'''


def get_osworld_tool() -> dict:
    """Get the OSWorld tool definition for LLM function calling.
    
    Returns:
        Tool definition dictionary with function schema
    """
    return {
        'type': 'function',
        'function': {
            'name': OSWORLD_TOOL_NAME,
            'description': OSWORLD_TOOLS_DESCRIPTION,
            'parameters': {
                'type': 'object',
                'properties': {
                    'method': {
                        'type': 'string',
                        'description': 'The OSWorld method to call',
                        'enum': [
                            'execute_action',
                            'get_screenshot',
                            'get_accessibility_tree',
                            'get_terminal_output',
                            'get_file',
                            'execute_python_command',
                            'run_python_script',
                            'run_bash_script',
                            'start_recording',
                            'end_recording',
                            'get_vm_platform',
                            'get_vm_screen_size',
                            'get_vm_window_size',
                            'get_vm_wallpaper',
                            'get_vm_desktop_path',
                            'get_vm_directory_tree',
                        ],
                    },
                    'params': {
                        'type': 'object',
                        'description': 'Parameters for the method (varies by method)',
                        'additionalProperties': True,
                    },
                },
                'required': ['method'],
            },
        },
    }
