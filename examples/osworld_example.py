#!/usr/bin/env python3
"""
Example script demonstrating OSWorld runtime usage with OSWorldInteractiveAction.

This script shows how to:
1. Configure and create an OSWorld runtime
2. Connect to a VM
3. Use OSWorldInteractiveAction to interact with the VM
4. Perform GUI actions (click, type, press keys)
5. Take screenshots
6. Get VM information (platform, screen size)
7. Clean up resources

Prerequisites:
- Singularity/Apptainer installed
- QEMU and OVMF installed
- VM image at ./OS_images/Ubuntu.qcow2 with OSWorld server
"""

import asyncio
import sys
from pathlib import Path

from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.singularity.osworld_singularity_runtime import (
    OSWorldSingularityRuntime,
)
from openhands.storage import get_file_store


async def main():
    """Main example function."""
    
    print("=" * 80)
    print("OSWorld Runtime Example")
    print("=" * 80)
    print()
    
    # 1. Create configuration
    print("1. Creating configuration...")
    config = OpenHandsConfig()
    config.runtime = 'osworld'
    config.sandbox.base_container_image = 'ubuntu:24.04'
    config.sandbox.run_as_fakeroot = True
    
    # Check if VM image exists
    vm_image_path = './OS_images/Ubuntu.qcow2'
    if not Path(vm_image_path).exists():
        print(f"ERROR: VM image not found at {vm_image_path}")
        print("Please place your Ubuntu VM image with OSWorld server at this location.")
        return 1
    
    print(f"✓ Configuration created")
    print(f"  Runtime: {config.runtime}")
    print(f"  Base image: {config.sandbox.base_container_image}")
    print(f"  VM image: {vm_image_path}")
    print()
    
    # 2. Create event stream
    print("2. Creating event stream...")
    file_store = get_file_store('local', '/tmp/osworld_example')
    event_stream = EventStream(sid='osworld-example', file_store=file_store)
    print("✓ Event stream created")
    print()
    
    # 3. Create OSWorld runtime
    print("3. Creating OSWorld runtime...")
    runtime = OSWorldSingularityRuntime(
        config=config,
        event_stream=event_stream,
        sid='osworld-example',
        os_type='linux',
        vm_image_path=vm_image_path,
        attach_to_existing=False,
    )
    print("✓ Runtime created")
    print(f"  OS Type: {runtime.os_type}")
    print()
    
    try:
        # 4. Connect to runtime (starts VM)
        print("4. Connecting to runtime (this may take 1-2 minutes)...")
        print("   - Building Singularity image (if needed)")
        print("   - Starting QEMU VM")
        print("   - Waiting for VM to boot")
        print("   - Waiting for OSWorld server to start")
        await runtime.connect()
        print("✓ Runtime connected and VM is ready!")
        print(f"  VM Server URL: {runtime.osworld_vm_url}")
        print(f"  VNC URL: {runtime.vnc_url}")
        print()
        
        # 5. Check if VM is alive
        print("5. Checking VM health...")
        runtime.check_if_alive()
        print("✓ VM is alive and responding")
        print()

         # Wait a bit
        await asyncio.sleep(10)
        
        # 6. Get VM screenshot using OSWorldInteractiveAction
        print("6. Taking VM screenshot...")
        action = OSWorldInteractiveAction(
            method='get_screenshot',
            params={},
            thought='Taking initial screenshot of the VM desktop'
        )
        observation = runtime.run_action(action)
        print(f"   Observation: {observation.content[:100]}...")  # Show first 100 chars
        
        # Also get raw screenshot for saving
        screenshot = runtime.get_vm_screenshot()
        if screenshot:
            screenshot_path = '/tmp/osworld_screenshot.png'
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            print(f"✓ Screenshot saved to {screenshot_path}")
            print(f"  Size: {len(screenshot)} bytes")
        else:
            print("✗ Failed to get screenshot")
        print()
        
        # 7. Execute some actions using OSWorldInteractiveAction
        print("7. Executing VM actions...")
        
        # Example 1: Click at position
        print("   a. Clicking at position (10, 10)...")
        action = OSWorldInteractiveAction(
            method='execute_action',
            params={
                'action': {
                    'action_type': 'CLICK',
                    'parameters': {'x': 10, 'y': 10, 'button': 'left'}
                }
            },
            thought='Clicking at center-ish position on the screen'
        )
        observation = runtime.run_action(action)
        print(f"      Result: {observation.content}")
        print(f"      Exit code: {observation.exit_code}")
        
        # Wait a bit
        await asyncio.sleep(1)
        
        # Example 2: Type some text
        print("   b. Typing 'Hello OSWorld'...")
        action = OSWorldInteractiveAction(
            method='execute_action',
            params={
                'action': {
                    'action_type': 'TYPING',
                    'parameters': {'text': 'Hello OSWorld'}
                }
            },
            thought='Typing a greeting message'
        )
        observation = runtime.run_action(action)
        print(f"      Result: {observation.content}")
        print(f"      Exit code: {observation.exit_code}")
        
        # Wait a bit
        await asyncio.sleep(1)
        
        # Example 3: Press Enter
        print("   c. Pressing Enter key...")
        action = OSWorldInteractiveAction(
            method='execute_action',
            params={
                'action': {
                    'action_type': 'PRESS',
                    'parameters': {'key': 'enter'}
                }
            },
            thought='Pressing Enter to confirm'
        )
        observation = runtime.run_action(action)
        print(f"      Result: {observation.content}")
        print(f"      Exit code: {observation.exit_code}")
        
        print("✓ Actions executed successfully")
        print()
        
        # 8. Get VM information
        print("8. Getting VM information...")
        
        # Get platform
        print("   a. Getting VM platform...")
        action = OSWorldInteractiveAction(
            method='get_vm_platform',
            params={},
            thought='Getting the operating system platform'
        )
        observation = runtime.run_action(action)
        print(f"      Platform: {observation.content}")
        
        # Get screen size
        print("   b. Getting screen size...")
        action = OSWorldInteractiveAction(
            method='get_vm_screen_size',
            params={},
            thought='Getting the screen dimensions'
        )
        observation = runtime.run_action(action)
        print(f"      Screen size: {observation.content}")
        
        print("✓ VM information retrieved")
        print()
        
        # 9. Test advanced OSWorld methods
        print("9. Testing advanced OSWorld methods...")
        
        # Test get_accessibility_tree
        print("   a. Getting accessibility tree...")
        action = OSWorldInteractiveAction(
            method='get_accessibility_tree',
            params={},
            thought='Getting UI accessibility tree for element inspection'
        )
        observation = runtime.run_action(action)
        if observation.content and len(observation.content) > 0:
            print(f"      Accessibility tree retrieved ({len(observation.content)} chars)")
            # Show first 200 characters
            print(f"      Preview: {observation.content[:200]}...")
        else:
            print("      Note: Accessibility tree not available or empty")
        
        await asyncio.sleep(1)
        
        # Test get_terminal_output
        print("   b. Getting terminal output...")
        action = OSWorldInteractiveAction(
            method='get_terminal_output',
            params={},
            thought='Getting terminal output from the VM'
        )
        observation = runtime.run_action(action)
        if observation.content and len(observation.content) > 0:
            print(f"      Terminal output retrieved ({len(observation.content)} chars)")
            print(f"      Preview: {observation.content[:200]}...")
        else:
            print("      Note: No terminal output available")
        
        await asyncio.sleep(1)
        
        # Test execute_python_command
        print("   c. Executing Python command...")
        action = OSWorldInteractiveAction(
            method='execute_python_command',
            params={
                'command': "print('Hello from Python!'); import sys; print(f'Python version: {sys.version}')"
            },
            thought='Running a simple Python command in the VM'
        )
        observation = runtime.run_action(action)
        print(f"      Python output: {observation.content}")
        print(f"      Exit code: {observation.exit_code}")
        
        await asyncio.sleep(1)
        
        # Test run_python_script
        print("   d. Running Python script...")
        python_script = """
import os
import platform

print(f"Hostname: {platform.node()}")
print(f"Python: {platform.python_version()}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Current directory: {os.getcwd()}")
print(f"Home directory: {os.path.expanduser('~')}")
"""
        action = OSWorldInteractiveAction(
            method='run_python_script',
            params={
                'script': python_script
            },
            thought='Running a multi-line Python script to get system info'
        )
        observation = runtime.run_action(action)
        print(f"      Script output:")
        for line in observation.content.split('\n')[:10]:  # Show first 10 lines
            if line.strip():
                print(f"        {line}")
        print(f"      Exit code: {observation.exit_code}")
        
        await asyncio.sleep(1)
        
        # Test run_bash_script
        print("   e. Running bash script...")
        bash_script = """echo "Hello from Bash!"
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"
"""
        action = OSWorldInteractiveAction(
            method='run_bash_script',
            params={
                'script': bash_script,
                'timeout': 30
            },
            thought='Running a simple bash script'
        )
        observation = runtime.run_action(action)
        
        # Handle both success and error cases
        from openhands.events.observation import ErrorObservation
        if isinstance(observation, ErrorObservation):
            print(f"      ⚠ Bash script error: {observation.content}")
        else:
            print(f"      Bash output:")
            for line in observation.content.split('\n'):
                if line.strip():
                    print(f"        {line}")
            if hasattr(observation, 'exit_code'):
                print(f"      Exit code: {observation.exit_code}")
        
        print("✓ Advanced methods tested")
        print()
        
        # 10. Test file download with get_file
        print("10. Testing file download (get_file)...")
        
        # First, create a test file in the VM
        print("   a. Creating test file in VM...")
        test_content = "Hello from OSWorld VM!\nThis is a test file.\nCreated at: $(date)"
        action = OSWorldInteractiveAction(
            method='run_bash_script',
            params={
                'script': f'echo "{test_content}" > /tmp/test_file.txt && cat /tmp/test_file.txt',
                'timeout': 10
            },
            thought='Creating a test file for download'
        )
        observation = runtime.run_action(action)
        if isinstance(observation, ErrorObservation):
            print(f"      ⚠ Could not create test file: {observation.content}")
        else:
            print(f"      Test file created")
        
        await asyncio.sleep(1)
        
        # Now download it using get_file
        print("   b. Downloading test file using get_file...")
        action = OSWorldInteractiveAction(
            method='get_file',
            params={'file_path': '/tmp/test_file.txt'},
            thought='Downloading test file from VM'
        )
        observation = runtime.run_action(action)
        if isinstance(observation, ErrorObservation):
            print(f"      ⚠ Failed to download: {observation.content}")
        else:
            try:
                import base64
                # Parse "base64:<data>" format
                if observation.content.startswith('base64:'):
                    content_b64 = observation.content[7:]  # Remove "base64:" prefix
                    file_data = base64.b64decode(content_b64)
                    download_path = '/tmp/osworld_downloaded_file.txt'
                    with open(download_path, 'wb') as f:
                        f.write(file_data)
                    print(f"      ✓ File downloaded to {download_path} ({len(file_data)} bytes)")
                    print(f"      Content preview: {file_data.decode('utf-8')[:100]}")
                else:
                    print(f"      Unexpected format: {observation.content[:100]}")
            except Exception as e:
                print(f"      Could not save file: {e}")
        
        print("✓ File download tested")
        print()
        
        # 11. Test VM information methods
        print("11. Testing VM information methods...")
        
        # Test get_vm_window_size
        print("   a. Getting VM window size...")
        action = OSWorldInteractiveAction(
            method='get_vm_window_size',
            params={
                'app_class_name': 'gnome-terminal-server'  # Common terminal app on Ubuntu
            },
            thought='Getting window size for a specific application'
        )
        observation = runtime.run_action(action)
        from openhands.events.observation import ErrorObservation
        if isinstance(observation, ErrorObservation):
            print(f"      Note: {observation.content}")
        else:
            print(f"      Window size: {observation.content}")
        
        await asyncio.sleep(1)
        
        # Test get_vm_wallpaper
        print("   b. Getting VM wallpaper...")
        action = OSWorldInteractiveAction(
            method='get_vm_wallpaper',
            params={},
            thought='Getting the desktop wallpaper image'
        )
        observation = runtime.run_action(action)
        if isinstance(observation, ErrorObservation):
            print(f"      Note: {observation.content}")
        else:
            try:
                import base64
                # The wallpaper endpoint returns the image directly as base64
                if observation.content.startswith('base64:'):
                    content_b64 = observation.content[7:]  # Remove "base64:" prefix
                    wallpaper_data = base64.b64decode(content_b64)
                    wallpaper_path = '/tmp/osworld_wallpaper.png'
                    with open(wallpaper_path, 'wb') as f:
                        f.write(wallpaper_data)
                    print(f"      ✓ Wallpaper saved to {wallpaper_path} ({len(wallpaper_data)} bytes)")
                else:
                    print(f"      Unexpected format: {observation.content[:100]}")
            except Exception as e:
                print(f"      Could not save wallpaper: {e}")
        
        await asyncio.sleep(1)
        
        # Test get_vm_desktop_path
        print("   c. Getting VM desktop path...")
        action = OSWorldInteractiveAction(
            method='get_vm_desktop_path',
            params={},
            thought='Getting the desktop directory path'
        )
        observation = runtime.run_action(action)
        desktop_path = observation.content
        print(f"      Desktop path: {desktop_path}")
        
        await asyncio.sleep(1)
        
        # Test get_vm_directory_tree
        print("   d. Getting VM directory tree...")
        action = OSWorldInteractiveAction(
            method='get_vm_directory_tree',
            params={
                'path': desktop_path if desktop_path else '/home'
            },
            thought='Listing directory contents'
        )
        observation = runtime.run_action(action)
        if isinstance(observation, ErrorObservation):
            print(f"      Error: {observation.content}")
        else:
            print(f"      Directory tree:")
            # Show first 10 lines
            for i, line in enumerate(observation.content.split('\n')[:10]):
                if line.strip():
                    print(f"        {line}")
            if len(observation.content.split('\n')) > 10:
                print(f"        ... ({len(observation.content.split('\n')) - 10} more lines)")
        
        print("✓ VM information methods tested")
        print()
        
        # 12. Test screen recording
        print("12. Testing screen recording...")
        
        # Start recording
        print("   a. Starting screen recording...")
        action = OSWorldInteractiveAction(
            method='start_recording',
            params={},
            thought='Starting to record the VM screen'
        )
        observation = runtime.run_action(action)
        if isinstance(observation, ErrorObservation):
            print(f"      ⚠ Recording not available: {observation.content}")
        else:
            print(f"      Recording started: {observation.content}")
            
            # Wait a few seconds while recording
            print("   b. Recording for 3 seconds...")
            await asyncio.sleep(3)
            
            # Do some actions while recording
            print("   c. Performing actions while recording...")
            action = OSWorldInteractiveAction(
                method='execute_action',
                params={
                    'action': {
                        'action_type': 'MOVE_TO',  # Correct: MOVE_TO, not MOVE
                        'parameters': {'x': 100, 'y': 100}
                    }
                },
                thought='Moving mouse to top-left during recording'
            )
            runtime.run_action(action)
            await asyncio.sleep(1)

            print("   d. Moving to center...")
            action = OSWorldInteractiveAction(
                method='execute_action',
                params={
                    'action': {
                        'action_type': 'MOVE_TO',  # Correct: MOVE_TO, not MOVE
                        'parameters': {'x': 512, 'y': 384}  # Center of 1024x768 screen
                    }
                },
                thought='Moving mouse to center during recording'
            )
            runtime.run_action(action)
            await asyncio.sleep(1)
            
            print("   e. Clicking at center...")
            action = OSWorldInteractiveAction(
                method='execute_action',
                params={
                    'action': {
                        'action_type': 'CLICK',
                        'parameters': {'x': 512, 'y': 384}
                    }
                },
                thought='Clicking at center during recording'
            )
            runtime.run_action(action)
            await asyncio.sleep(1)
            
            # Stop recording
            print("   f. Stopping recording and downloading...")
            action = OSWorldInteractiveAction(
                method='end_recording',
                params={},  # Note: 'dest' parameter is ignored by OSWorld server
                thought='Stopping the screen recording'
            )
            observation = runtime.run_action(action)
            if isinstance(observation, ErrorObservation):
                print(f"      ⚠ Failed to stop recording: {observation.content}")
            else:
                try:
                    import base64
                    # The end_recording endpoint returns the video directly as base64
                    if observation.content.startswith('base64:'):
                        content_b64 = observation.content[7:]  # Remove "base64:" prefix
                        video_data = base64.b64decode(content_b64)
                        video_path = '/tmp/osworld_recording.mp4'
                        with open(video_path, 'wb') as f:
                            f.write(video_data)
                        print(f"      ✓ Recording saved to {video_path} ({len(video_data)} bytes)")
                    else:
                        print(f"      Unexpected format: {observation.content[:100]}")
                except Exception as e:
                    print(f"      Could not save recording: {e}")
        
        print("✓ Screen recording tested")
        print()
        
        # 13. Get another screenshot to see changes
        print("13. Taking final screenshot...")
        action = OSWorldInteractiveAction(
            method='get_screenshot',
            params={},
            thought='Taking final screenshot after interactions'
        )
        observation = runtime.run_action(action)
        
        screenshot = runtime.get_vm_screenshot()
        if screenshot:
            screenshot_path = '/tmp/osworld_screenshot_after.png'
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            print(f"✓ Final screenshot saved to {screenshot_path}")

        # Wait a bit
        await asyncio.sleep(2)
 
        print()
        
        print("=" * 80)
        print("Example completed successfully!")
        print("=" * 80)
        print()
        print("You can:")
        print(f"1. View screenshots: open /tmp/osworld_screenshot*.png")
        print(f"2. View wallpaper: open /tmp/osworld_wallpaper.png")
        print(f"3. View recording: vlc /tmp/osworld_recording.mp4")
        print(f"4. View downloaded file: cat /tmp/osworld_downloaded_file.txt")
        print(f"5. Connect via VNC: {runtime.vnc_url}")
        print(f"6. Access VM API: curl {runtime.osworld_vm_url}/screenshot")
        print()
        
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.exception("Failed to run example")
        return 1
        
    finally:
        # 14. Clean up
        print("14. Cleaning up...")
        runtime.close()
        print("✓ Runtime closed")
        print()


if __name__ == '__main__':
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

