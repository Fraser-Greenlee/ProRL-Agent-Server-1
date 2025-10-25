#!/usr/bin/env python3
import os
import shutil
import tempfile
import time

from openhands.core.config import OpenHandsConfig, SandboxConfig
from openhands.events.action import OSInteractiveAction
from openhands.events.observation import CmdOutputObservation
from openhands.events.stream import EventStream
from openhands.runtime.impl.singularity.singularity_runtime import SingularityRuntime
from openhands.storage import get_file_store

try:
    from PIL import Image
except Exception:
    Image = None


def demonstrate_computer_use_tool(
    runtime: SingularityRuntime, tool_name: str, action_command: str, description: str
):
    """Demonstrate a single computer_use tool with detailed input/output."""
    print(f'\n{"=" * 60}')
    print(f'🧪 TESTING: {tool_name}')
    print(f'📝 Description: {description}')
    print(f'{"=" * 60}')

    # Create action
    action = OSInteractiveAction(
        os_actions=action_command, thought=f'Testing {tool_name}: {description}'
    )

    # Show input
    print('📥 INPUT:')
    print('   Action Type: OSInteractiveAction')
    print(f'   OS Actions: {action.os_actions}')
    print(f'   Thought: {action.thought}')

    # Execute action
    print('\n⚡ EXECUTING...')
    action.set_hard_timeout(30000)
    obs = runtime.run_action(action)

    # Show output
    print('\n📤 OUTPUT:')
    assert isinstance(obs, CmdOutputObservation)

    print(f'   Success: {"✅ Yes" if obs.exit_code == 0 else "❌ No"}')
    print(f'   Exit Code: {obs.exit_code}')
    print(f'   Content:\n{obs.content}')
    print(f'   Command: {obs.command}')

    return obs


def _cleanup_old_workspaces(prefix: str, keep_path: str | None) -> None:
    """
    Remove old temporary workspaces under /tmp matching the given prefix,
    except for 'keep_path' if provided.
    """
    try:
        tmp_dir = '/tmp'
        for name in os.listdir(tmp_dir):
            if not name.startswith(prefix):
                continue
            full = os.path.join(tmp_dir, name)
            if keep_path and os.path.abspath(full) == os.path.abspath(keep_path):
                continue
            if os.path.isdir(full):
                try:
                    shutil.rmtree(full, ignore_errors=True)
                except Exception:
                    pass
    except Exception:
        pass


def _save_screenshots_gif(
    workspace_dir: str,
    output_dir: str,
    gif_name: str = 'computer_use_run.gif',
    duration_ms: int = 1000,
    image_prefix: str = 'computer_screenshot_',
) -> str | None:
    """Create a GIF from screenshots taken during computer_use tests."""
    try:
        screenshots_dir = os.path.join(workspace_dir, '.computer_screenshots')
        if not os.path.isdir(screenshots_dir):
            return None

        # Filter files by prefix and image extensions
        all_files = [
            f
            for f in os.listdir(screenshots_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        files = [f for f in all_files if f.startswith(image_prefix)]

        if not files:
            return None
        files.sort()  # filenames contain timestamps; lexicographic order works

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, gif_name)

        if Image is None:
            # PIL not available; skip silently
            return None

        frames = []
        for fname in files:
            p = os.path.join(screenshots_dir, fname)
            try:
                img = Image.open(p).convert('RGB')
                frames.append(img)
            except Exception:
                continue
        if not frames:
            return None

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            quality=85,
            format='GIF',
        )
        return output_path
    except Exception:
        return None


def main():
    """Main demonstration function."""
    print('🚀 Computer Use Tools Demonstration with Singularity Environment')
    print('=' * 60)

    # Setup workspace (clean up old runs first)
    workspace_dir = tempfile.mkdtemp(prefix='computer_use_demo_')
    singularity_image_path = '/root/singularity_images/oh_v0.40.0_ubuntu_t_24.04.sif'

    print(f'📁 Workspace: {workspace_dir}')
    print(f'🖼️ Singularity Image: {singularity_image_path}')

    # Setup Singularity runtime
    sandbox_config = SandboxConfig(
        runtime_container_image=singularity_image_path,
        browsergym_eval_env=None,
        timeout=30,
        use_host_network=True,
    )
    sandbox_config.volumes = '/work/Projects/ProRL-Agent-Server/:/openhands/code:ro'

    config = OpenHandsConfig(
        default_agent='CodeActAgent',
        runtime='singularity',
        sandbox=sandbox_config,
        workspace_base=workspace_dir,
        workspace_mount_path=workspace_dir,
        run_as_openhands=False,  # This prevents the UID 0 conflict
    )

    # Setup event stream (required for SingularityRuntime)
    file_store = get_file_store('local', workspace_dir)
    event_stream = EventStream('computer_use_test_session', file_store)

    runtime = SingularityRuntime(
        config=config,
        event_stream=event_stream,
        sid='computer_use_test_session',
        plugins=[],
        headless_mode=True,
    )

    try:
        print('\n🔌 Connecting to runtime...')
        # Connect to the runtime (this starts the Singularity container and action server)
        import asyncio

        # Check if Singularity image exists
        if not os.path.exists(singularity_image_path):
            raise FileNotFoundError(
                f'Singularity image not found: {singularity_image_path}'
            )

        asyncio.run(runtime.connect())
        print('✅ Runtime connected successfully')
        print('🖥️ Xvfb and xfce4 will be started on first computer_use action')

        # Create screenshots directory
        screenshots_dir = os.path.join(workspace_dir, '.computer_screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)

        # Test 1: Get screen size
        demonstrate_computer_use_tool(
            runtime,
            'size()',
            'size()',
            'Get the screen dimensions',
        )

        # Test 2: Get mouse position
        demonstrate_computer_use_tool(
            runtime,
            'position()',
            'position()',
            'Get current mouse cursor position',
        )

        # Test 3: Move mouse to specific position
        demonstrate_computer_use_tool(
            runtime,
            'moveTo(x, y)',
            'moveTo(200, 200)',
            'Move mouse cursor to absolute position (200, 200)',
        )

        # Test 4: Move mouse relative to current position
        demonstrate_computer_use_tool(
            runtime,
            'move(xOffset, yOffset)',
            'move(100, 50)',
            'Move mouse cursor relative by (100, 50) pixels',
        )

        # Test 5: Click at current position
        demonstrate_computer_use_tool(
            runtime,
            'click()',
            'click()',
            'Click left mouse button at current position',
        )

        # Test 6: Click at specific position
        demonstrate_computer_use_tool(
            runtime,
            'click(x, y)',
            'click(400, 300)',
            'Click left mouse button at position (400, 300)',
        )

        # Test 7: Right click
        demonstrate_computer_use_tool(
            runtime,
            'rightClick()',
            'rightClick()',
            'Right click at current position',
        )

        # Test 8: Double click
        demonstrate_computer_use_tool(
            runtime,
            'doubleClick()',
            'doubleClick(500, 400)',
            'Double click at position (500, 400)',
        )

        # Test 9: Take a screenshot
        screenshot_path = os.path.join(
            screenshots_dir, f'computer_screenshot_{int(time.time())}_01.png'
        )
        demonstrate_computer_use_tool(
            runtime,
            'screenshot(path)',
            f'screenshot("{screenshot_path}")',
            'Capture a screenshot of the desktop',
        )

        # Test 10: Keyboard - type text
        demonstrate_computer_use_tool(
            runtime,
            'typewrite(message)',
            'typewrite("Hello from PyAutoGUI!")',
            'Type text using keyboard',
        )

        # Test 11: Press Enter key
        demonstrate_computer_use_tool(
            runtime,
            'press(key)',
            'press("enter")',
            'Press the Enter key',
        )

        # Test 12: Press multiple keys (hotkey)
        demonstrate_computer_use_tool(
            runtime,
            'hotkey(*keys)',
            'hotkey("ctrl", "a")',
            'Press Ctrl+A hotkey combination',
        )

        # Test 13: Scroll up
        demonstrate_computer_use_tool(
            runtime,
            'scroll(clicks)',
            'scroll(5)',
            'Scroll mouse wheel up (positive value)',
        )

        # Test 14: Scroll down
        demonstrate_computer_use_tool(
            runtime,
            'scroll(clicks)',
            'scroll(-3)',
            'Scroll mouse wheel down (negative value)',
        )

        # Test 15: Complex multi-line action
        screenshot_path_2 = os.path.join(
            screenshots_dir, f'computer_screenshot_{int(time.time())}_02.png'
        )
        multi_action = f"""moveTo(100, 100)
click()
typewrite("Multi-line test")
press("enter")
screenshot("{screenshot_path_2}")"""

        demonstrate_computer_use_tool(
            runtime,
            'Multi-Action Sequence',
            multi_action,
            'Execute multiple PyAutoGUI actions in sequence',
        )

        # Test 16: Test with wait/sleep
        screenshot_path_3 = os.path.join(
            screenshots_dir, f'computer_screenshot_{int(time.time())}_03.png'
        )
        timed_action = f"""moveTo(600, 400)
sleep(0.5)
click()
sleep(0.3)
screenshot("{screenshot_path_3}")"""

        demonstrate_computer_use_tool(
            runtime,
            'Timed Actions',
            timed_action,
            'Execute actions with sleep delays between them',
        )

        # Test 17: Drag operation
        demonstrate_computer_use_tool(
            runtime,
            'drag(x, y, duration)',
            'drag(700, 500, duration=1.0)',
            'Drag mouse from current position to (700, 500) over 1 second',
        )

        # Test 18: Key press and release
        key_action = """keyDown("shift")
typewrite("hello world")
keyUp("shift")"""

        demonstrate_computer_use_tool(
            runtime,
            'Key Down/Up',
            key_action,
            'Hold Shift key down, type, then release',
        )

        # Test 19: Final screenshot
        screenshot_path_4 = os.path.join(
            screenshots_dir, f'computer_screenshot_{int(time.time())}_04.png'
        )
        demonstrate_computer_use_tool(
            runtime,
            'screenshot(path)',
            f'screenshot("{screenshot_path_4}")',
            'Capture final screenshot',
        )

        print('\n🎉 Computer Use Tools Demonstration Completed!')
        print(
            '📊 Demonstrated 19 different computer_use operations with detailed input/output'
        )
        print(f'📁 Test files available in: {workspace_dir}')
        print(f'📸 Screenshots saved in: {screenshots_dir}')

        # Save run GIF into logs directory
        logs_dir = '/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/logs'

        # Create GIF from screenshots
        gif_path = _save_screenshots_gif(
            workspace_dir,
            logs_dir,
            gif_name='computer_use_run.gif',
            duration_ms=1000,
            image_prefix='computer_screenshot_',
        )
        if gif_path:
            print(f'🎞️ Saved Computer Use Screenshots GIF: {gif_path}')
        else:
            print('⚠️ No screenshots GIF saved (no screenshots or PIL not available)')

        # List all screenshots created
        if os.path.isdir(screenshots_dir):
            screenshot_files = [
                f
                for f in os.listdir(screenshots_dir)
                if f.startswith('computer_screenshot_')
            ]
            if screenshot_files:
                print(f'\n📸 Screenshots created ({len(screenshot_files)}):')
                for f in sorted(screenshot_files):
                    full_path = os.path.join(screenshots_dir, f)
                    size = os.path.getsize(full_path)
                    print(f'   - {f} ({size:,} bytes)')

    except Exception as e:
        print(f'\n❌ Demonstration failed: {str(e)}')
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print('\n🧹 Cleanup completed')
        print(f'📁 Workspace directory: {workspace_dir}')
        print('   (Not deleted for inspection - remove manually if needed)')


if __name__ == '__main__':
    main()
