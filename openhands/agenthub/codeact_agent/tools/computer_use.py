from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import COMPUTER_USE_TOOL_NAME

_COMPUTER_USE_DESCRIPTION = """Interact with the operating system using Python code. Use it when you need to control the mouse, keyboard, or take screenshots.

See the description of "code" parameter for more details.

Multiple actions can be provided at once, but will be executed sequentially without any feedback.
More than 2-3 actions usually leads to failure or unexpected behavior. Example:
moveTo(100, 200)
click()
typewrite('Hello World')
press('enter')
"""

_COMPUTER_USE_TOOL_DESCRIPTION = """
The following functions are available from pyautogui. Nothing else is supported.

# Mouse Control Functions

moveTo(x: int, y: int, duration: float = 0.0)
    Description: Move the mouse cursor to the specified (x, y) position on screen.
    Args:
        x: X coordinate (0 is left edge)
        y: Y coordinate (0 is top edge)
        duration: Time in seconds to move the mouse (default: instant)
    Examples:
        moveTo(100, 200)
        moveTo(500, 500, duration=2.0)

move(xOffset: int, yOffset: int, duration: float = 0.0)
    Description: Move the mouse cursor relative to its current position.
    Args:
        xOffset: Number of pixels to move horizontally (positive = right, negative = left)
        yOffset: Number of pixels to move vertically (positive = down, negative = up)
        duration: Time in seconds to move the mouse
    Examples:
        move(50, 0)
        move(-100, -100, duration=1.0)

click(x: int | None = None, y: int | None = None, clicks: int = 1, interval: float = 0.0, button: str = 'left')
    Description: Click the mouse button at current position or specified position.
    Args:
        x: Optional X coordinate to click at
        y: Optional Y coordinate to click at
        clicks: Number of clicks (default: 1)
        interval: Seconds between clicks (default: 0.0)
        button: Which button to click - 'left', 'middle', or 'right' (default: 'left')
    Examples:
        click()
        click(100, 200)
        click(clicks=2, interval=0.25)
        click(button='right')

doubleClick(x: int | None = None, y: int | None = None, interval: float = 0.0, button: str = 'left')
    Description: Double click the mouse button.
    Args:
        x: Optional X coordinate
        y: Optional Y coordinate
        interval: Seconds between clicks
        button: Which button - 'left', 'middle', or 'right'
    Examples:
        doubleClick()
        doubleClick(100, 200)

rightClick(x: int | None = None, y: int | None = None)
    Description: Right click the mouse.
    Examples:
        rightClick()
        rightClick(500, 300)

drag(x: int, y: int, duration: float = 0.0, button: str = 'left')
    Description: Drag mouse to a position while holding down button.
    Args:
        x: X coordinate to drag to
        y: Y coordinate to drag to
        duration: Time in seconds to perform drag
        button: Which button to hold - 'left', 'middle', or 'right'
    Examples:
        drag(100, 200)
        drag(300, 400, duration=2.0, button='right')

scroll(clicks: int, x: int | None = None, y: int | None = None)
    Description: Scroll the mouse wheel.
    Args:
        clicks: Amount to scroll (positive = up, negative = down)
        x: Optional X coordinate to scroll at
        y: Optional Y coordinate to scroll at
    Examples:
        scroll(10)
        scroll(-3)
        scroll(5, x=500, y=500)

position()
    Description: Get the current mouse position as a tuple (x, y).
    Examples:
        position()

# Keyboard Control Functions

typewrite(message: str, interval: float = 0.0)
    Description: Type a string of characters.
    Args:
        message: The string to type
        interval: Seconds between each keypress (default: 0.0)
    Examples:
        typewrite('Hello World')
        typewrite('Slow typing', interval=0.1)

write(message: str, interval: float = 0.0)
    Description: Alias for typewrite. Type a string of characters.
    Examples:
        write('Hello World')

press(keys: str | list[str], presses: int = 1, interval: float = 0.0)
    Description: Press a key or list of keys.
    Args:
        keys: Key name or list of key names to press
        presses: Number of times to press (default: 1)
        interval: Seconds between presses
    Valid key names include: 'enter', 'esc', 'tab', 'space', 'backspace', 'delete', 'up', 'down',
    'left', 'right', 'home', 'end', 'pageup', 'pagedown', 'f1'-'f12', 'shift', 'ctrl', 'alt',
    'win', 'command', 'option', 'a'-'z', '0'-'9', etc.
    Examples:
        press('enter')
        press(['ctrl', 'c'])
        press('a', presses=3)

keyDown(key: str)
    Description: Press and hold a key down.
    Examples:
        keyDown('shift')
        keyDown('ctrl')

keyUp(key: str)
    Description: Release a key.
    Examples:
        keyUp('shift')
        keyUp('ctrl')

hotkey(*keys: str)
    Description: Press multiple keys simultaneously (like Ctrl+C).
    Args:
        *keys: Variable number of key names
    Examples:
        hotkey('ctrl', 'c')
        hotkey('ctrl', 'alt', 'delete')
        hotkey('command', 'v')

# Screen Functions

size()
    Description: Get the screen size as a tuple (width, height).
    Examples:
        size()

screenshot(imageFilename: str | None = None, region: tuple[int, int, int, int] | None = None)
    Description: Take a screenshot and optionally save it.
    Args:
        imageFilename: Optional path to save the screenshot
        region: Optional (left, top, width, height) tuple to capture only part of screen
    Returns: PIL Image object
    Examples:
        screenshot()
        screenshot('/tmp/screenshot.png')
        screenshot(region=(0, 0, 300, 400))

locateOnScreen(image: str, confidence: float = 0.9)
    Description: Locate an image on the screen and return its position.
    Args:
        image: Path to the image file to locate
        confidence: Match confidence (0.0 to 1.0, default: 0.9)
    Returns: Box object with (left, top, width, height) or None if not found
    Examples:
        locateOnScreen('/tmp/button.png')
        locateOnScreen('/tmp/icon.png', confidence=0.8)

locateCenterOnScreen(image: str, confidence: float = 0.9)
    Description: Locate an image and return its center coordinates.
    Args:
        image: Path to the image file to locate
        confidence: Match confidence (0.0 to 1.0)
    Returns: Point object with (x, y) or None if not found
    Examples:
        locateCenterOnScreen('/tmp/button.png')

# Alert and Message Box Functions

alert(text: str = '', title: str = '', button: str = 'OK')
    Description: Display a simple alert box.
    Examples:
        alert('Task complete!')
        alert(text='Warning!', title='Alert', button='OK')

confirm(text: str = '', title: str = '', buttons: list[str] = ['OK', 'Cancel'])
    Description: Display a confirmation dialog.
    Returns: The text of the button clicked
    Examples:
        confirm('Are you sure?')
        confirm('Proceed?', title='Confirmation', buttons=['Yes', 'No'])

prompt(text: str = '', title: str = '', default: str = '')
    Description: Display a text input prompt.
    Returns: The text entered by the user or None if cancelled
    Examples:
        prompt('Enter your name:')
        prompt('Enter value:', default='100')

# Utility Functions

sleep(seconds: float)
    Description: Pause execution for a given number of seconds.
    Examples:
        sleep(1.0)
        sleep(0.5)
"""

ComputerUseTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=COMPUTER_USE_TOOL_NAME,
        description=_COMPUTER_USE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': (
                        'The Python code that interacts with the operating system using pyautogui.\n'
                        + _COMPUTER_USE_TOOL_DESCRIPTION
                    ),
                }
            },
            'required': ['code'],
        },
    ),
)

