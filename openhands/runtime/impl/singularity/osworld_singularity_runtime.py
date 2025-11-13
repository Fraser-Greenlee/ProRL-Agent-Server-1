"""OSWorld Singularity Runtime for running QEMU VMs in Singularity containers.

This runtime extends SingularityRuntime to support OSWorld environments by:
1. Running QEMU VMs inside Singularity containers
2. Managing port mappings for parallel VM instances
3. Communicating with OSWorld server running inside the VM
"""

import os
import subprocess
import signal
import time
import json
import threading
from pathlib import Path
from typing import Callable

import httpx

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.impl.singularity.singularity_runtime import (
    SingularityRuntime,
    CONTAINER_NAME_PREFIX,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.command import DEFAULT_MAIN_MODULE

# Port ranges for OSWorld VM services
OSWORLD_VM_SERVER_PORT_RANGE = (15000, 19999)  # OSWorld Flask server inside VM
OSWORLD_VNC_PORT_RANGE = (18000, 19999)  # VNC server for VM display

OSWORLD_CONTAINER_NAME_PREFIX = 'openhands-osworld-runtime-'


class OSWorldSingularityRuntime(SingularityRuntime):
    """Runtime for OSWorld environments using QEMU VMs in Singularity containers.
    
    This runtime manages QEMU VMs inside Singularity containers, providing:
    - Port management for parallel VM instances
    - Communication with OSWorld server inside the VM
    - Support for both Linux and Windows VMs
    """
    
    _osworld_port_allocation_lock = threading.Lock()
    
    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        main_module: str = DEFAULT_MAIN_MODULE,
        os_type: str = 'linux',  # 'linux' or 'windows'
        vm_image_path: str | None = None,
    ):
        """Initialize OSWorld Singularity Runtime.
        
        Args:
            config: OpenHands configuration
            event_stream: Event stream for communication
            sid: Session ID
            plugins: List of plugin requirements
            env_vars: Environment variables
            status_callback: Callback for status updates
            attach_to_existing: Whether to attach to existing container
            headless_mode: Whether to run in headless mode
            main_module: Main module to run
            os_type: Type of OS in VM ('linux' or 'windows')
            vm_image_path: Path to QCOW2 VM image file
        """
        self.os_type = os_type.lower()
        self.vm_image_path = vm_image_path or self._get_default_vm_image_path()
        self.qemu_process: subprocess.Popen | None = None
        self.qemu_pid: int | None = None
        self._qemu_stdout = None
        self._qemu_stderr = None
        self._vm_server_port: int = -1
        self._vnc_port: int = -1
        
        # Override container name prefix for OSWorld
        self.container_name = OSWORLD_CONTAINER_NAME_PREFIX + sid
        
        # Call parent constructor
        super().__init__(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
            main_module=main_module,
        )
        
    def _get_default_vm_image_path(self) -> str:
        """Get default VM image path based on OS type."""
        if self.os_type == 'linux':
            return '/OS_images/Ubuntu.qcow2'
        elif self.os_type == 'windows':
            return '/OS_images/Windows-10-x64.qcow2'
        else:
            raise ValueError(f'Unsupported OS type: {self.os_type}')
    
    def _get_singularity_image_path(self) -> str:
        """Get the Singularity image file path for OSWorld runtime."""
        # Use a specific OSWorld runtime image
        if self.runtime_container_image:
            if not self.runtime_container_image.endswith('.sif'):
                image_name = f'osworld_{self.runtime_container_image.replace(":", "_").replace("/", "_")}'
                from openhands.runtime.utils.singularity_runtime_build import get_runtime_image_repo
                image_repo = get_runtime_image_repo()
                os.makedirs(image_repo, exist_ok=True)
                return f'{image_repo}/{image_name}.sif'
            else:
                return self.runtime_container_image
        # Default to ubuntu base for OSWorld
        from openhands.runtime.utils.singularity_runtime_build import get_runtime_image_repo
        image_repo = get_runtime_image_repo()
        os.makedirs(image_repo, exist_ok=True)
        return f'{image_repo}/osworld_ubuntu_24_04.sif'
    
    def _allocate_osworld_ports(self) -> tuple[int, int]:
        """Allocate ports for OSWorld VM server and VNC.
        
        Returns:
            Tuple of (vm_server_port, vnc_port)
        """
        with OSWorldSingularityRuntime._osworld_port_allocation_lock:
            vm_server_port = find_available_tcp_port(
                OSWORLD_VM_SERVER_PORT_RANGE[0],
                OSWORLD_VM_SERVER_PORT_RANGE[1]
            )
            vnc_port = find_available_tcp_port(
                OSWORLD_VNC_PORT_RANGE[0],
                OSWORLD_VNC_PORT_RANGE[1]
            )
            return vm_server_port, vnc_port
    
    def _get_qemu_command(self) -> list[str]:
        """Build QEMU command based on OS type and configuration."""
        # Check if VM image exists
        if not os.path.exists(self.vm_image_path):
            raise FileNotFoundError(
                f'VM image not found: {self.vm_image_path}. '
                f'Please ensure the QCOW2 image file exists.'
            )
        
        # Get the VM image filename (it will be mounted at /OS_images/ inside container)
        vm_image_filename = os.path.basename(self.vm_image_path)
        vm_image_container_path = f'/OS_images/{vm_image_filename}'
        
        # Base QEMU command
        cmd = [
            'qemu-system-x86_64',
            '-bios', '/usr/share/ovmf/OVMF.fd',
            '-machine', 'q35',
            '-cpu', 'host',
            '-enable-kvm',
            '-m', '2G',
            '-smp', '2',
            '-drive', f'file={vm_image_container_path},if=ide',
            '-netdev', f'user,id=net0,hostfwd=tcp::{self._vm_server_port}-:5000,hostfwd=tcp::{self._vnc_port}-:8006',
            '-device', 'virtio-net-pci,netdev=net0',
            '-vnc', ':0',
            # Note: Don't use -daemonize, we manage the process with Popen
        ]
        
        return cmd
    
    def maybe_prepare_runtime_container_image(self):
        """Prepare the OSWorld runtime container image."""
        # Use simple template for OSWorld
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                # Default to Ubuntu 24.04 for OSWorld
                self.base_container_image = 'ubuntu:24.04'
            
            self.send_status_message('STATUS$STARTING_CONTAINER')
            
            with SingularityRuntime._runtime_builder_lock:
                from openhands.runtime.utils.singularity_runtime_build import (
                    build_runtime_image_from_template,
                )
                
                # Build OSWorld-specific image
                template_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../utils/runtime_templates/osworld_singularity.j2'
                )
                
                self.runtime_container_image = build_runtime_image_from_template(
                    base_image=self.base_container_image,
                    template_path=template_path,
                    runtime_builder=self.runtime_builder,
                    platform=self.config.sandbox.platform,
                    extra_deps=self.config.sandbox.runtime_extra_deps,
                    force_rebuild=self.config.sandbox.force_rebuild_runtime,
                )
        else:
            # Pull the image if it doesn't exist locally
            self._pull_image_if_needed()
    
    def init_container(self):
        """Initialize the Singularity container and start QEMU VM."""
        self.log('debug', 'Preparing to start OSWorld Singularity container with QEMU VM...')
        self.send_status_message('STATUS$PREPARING_CONTAINER')
        
        # Allocate ports for OSWorld services
        self._vm_server_port, self._vnc_port = self._allocate_osworld_ports()
        
        self.log(
            'info',
            f'Allocated OSWorld ports - VM Server: {self._vm_server_port}, VNC: {self._vnc_port}'
        )
        
        # Get the image path
        image_path = self._get_singularity_image_path()
        if not os.path.exists(image_path):
            raise RuntimeError(f'Singularity image not found: {image_path}')
        
        # Prepare environment variables for QEMU
        env_vars = {
            'VM_SERVER_PORT': str(self._vm_server_port),
            'VNC_PORT': str(self._vnc_port),
            'OS_TYPE': self.os_type,
        }
        
        # Get absolute path to VM image
        vm_image_abs_path = os.path.abspath(self.vm_image_path)
        vm_image_dir = os.path.dirname(vm_image_abs_path)
        
        # Build the singularity exec command to run QEMU
        cmd = [
            'singularity', 'exec',
            '--pid',
            '--writable-tmpfs',
            '--no-mount', 'home,cwd,tmp',
            '--home', '/root',
        ]
        
        # Add fakeroot if configured
        if self.config.sandbox.run_as_fakeroot:
            cmd.extend(['--fakeroot'])
        
        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(['--env', f'{key}={value}'])
        
        # Mount VM image directory (needs to be writable for QEMU to maintain disk state)
        cmd.extend(['--bind', f'{vm_image_dir}:/OS_images'])
        
        # Add image path
        cmd.append(image_path)
        
        # Add QEMU command
        qemu_cmd = self._get_qemu_command()
        cmd.extend(qemu_cmd)
        
        self.log('debug', f'Starting QEMU VM with command: {" ".join(cmd)}')
        
        try:
            # Create log directory for QEMU output
            log_dir = '/tmp/openhands_osworld_logs'
            os.makedirs(log_dir, exist_ok=True)
            qemu_stdout_path = os.path.join(log_dir, f'{self.sid}_qemu.out')
            qemu_stderr_path = os.path.join(log_dir, f'{self.sid}_qemu.err')
            
            # Start QEMU in the container (non-blocking with Popen)
            self._qemu_stdout = open(qemu_stdout_path, 'w')
            self._qemu_stderr = open(qemu_stderr_path, 'w')
            
            self.qemu_process = subprocess.Popen(
                cmd,
                stdout=self._qemu_stdout,
                stderr=self._qemu_stderr,
                text=True,
                start_new_session=True  # Create new process group for easier cleanup
            )
            
            # Save the QEMU PID
            self.qemu_pid = self.qemu_process.pid
            
            # Check if QEMU process started successfully
            time.sleep(2)  # Give QEMU a moment to start
            if self.qemu_process.poll() is not None:
                # Process failed to start
                self._qemu_stdout.close()
                self._qemu_stderr.close()
                with open(qemu_stderr_path, 'r') as f:
                    error_output = f.read()
                raise RuntimeError(
                    f'QEMU failed to start. Return code: {self.qemu_process.returncode}\n'
                    f'Error: {error_output}'
                )
            
            self.log('info', f'QEMU VM started with PID: {self.qemu_pid}')
            self.log('debug', f'QEMU logs: stdout={qemu_stdout_path}, stderr={qemu_stderr_path}')
            
            # Wait for VM to boot and OSWorld server to be ready
            self._wait_for_vm_ready()
            
            # Store session information
            session_info = {
                'vm_server_port': self._vm_server_port,
                'vnc_port': self._vnc_port,
                'os_type': self.os_type,
                'vm_image_path': self.vm_image_path,
                'qemu_pid': self.qemu_pid,
            }
            self._save_session_port_info(session_info)
            
            self.log('info', 'OSWorld VM is ready')
            self.send_status_message('STATUS$CONTAINER_STARTED')
            
        except Exception as e:
            self.log('error', f'Error starting OSWorld runtime: {str(e)}')
            self.close()
            raise e
    
    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for the VM to boot and OSWorld server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self.log('info', 'Waiting for OSWorld VM to boot...')
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to OSWorld server
                response = httpx.get(
                    f'http://localhost:{self._vm_server_port}/screenshot',
                    timeout=5.0
                )
                if response.status_code == 200:
                    self.log('info', 'OSWorld VM server is ready!')
                    return
            except Exception:
                pass
            
            # Check every 5 seconds
            time.sleep(5)
            self.log('debug', f'Still waiting for VM... ({int(time.time() - start_time)}s elapsed)')
        
        raise TimeoutError(
            f'OSWorld VM failed to become ready within {timeout} seconds. '
            f'VM Server port: {self._vm_server_port}'
        )
    
    def _is_container_running(self) -> bool:
        """Check if the QEMU VM is currently running."""
        # For OSWorld, we check the QEMU process instead of container process
        if self.qemu_process is not None:
            return self.qemu_process.poll() is None
        
        # If we attached to an existing QEMU, check by PID
        if self.qemu_pid is not None:
            try:
                os.kill(self.qemu_pid, 0)  # Check if process exists
                return True
            except (OSError, ProcessLookupError):
                return False
        
        return False
    
    def wait_until_alive(self):
        """Wait until the OSWorld VM is ready."""
        # For OSWorld, we only need to check if QEMU is running
        # The VM readiness check is already done in init_container
        if not self._is_container_running():
            raise AgentRuntimeDisconnectedError(
                f'QEMU VM for {self.container_name} is not running.'
            )
        # OSWorld VM is already validated in _wait_for_vm_ready()
        self.log('debug', 'OSWorld runtime is alive and ready')
    
    def _attach_to_container(self):
        """Attach to an existing OSWorld container."""
        # Get port information from session registry
        session_info = self._load_session_port_info()
        if not session_info:
            raise AgentRuntimeNotFoundError(
                f'OSWorld container {self.container_name} not found or not running.'
            )
        
        self._vm_server_port = session_info.get('vm_server_port', -1)
        self._vnc_port = session_info.get('vnc_port', -1)
        self.os_type = session_info.get('os_type', 'linux')
        self.vm_image_path = session_info.get('vm_image_path', self._get_default_vm_image_path())
        self.qemu_pid = session_info.get('qemu_pid')
        
        self.log(
            'debug',
            f'Attached to OSWorld container: {self.container_name} '
            f'VM Server: {self._vm_server_port}, VNC: {self._vnc_port}, QEMU PID: {self.qemu_pid}'
        )
    
    def check_if_alive(self) -> None:
        """Check if the OSWorld VM server is alive."""
        try:
            response = httpx.get(
                f'http://localhost:{self._vm_server_port}/screenshot',
                timeout=5.0
            )
            if response.status_code != 200:
                raise AgentRuntimeDisconnectedError('OSWorld VM server is not responding')
        except Exception as e:
            raise AgentRuntimeDisconnectedError(
                f'OSWorld VM server is not reachable: {str(e)}'
            )
    
    def close(self, rm_all_containers: bool | None = None):
        """Close the OSWorld runtime and stop QEMU VM."""
        # Close QEMU log file handles
        try:
            if self._qemu_stdout is not None:
                self._qemu_stdout.close()
                self._qemu_stdout = None
        except Exception as e:
            logger.warning(f'Failed to close QEMU stdout: {e}')
        try:
            if self._qemu_stderr is not None:
                self._qemu_stderr.close()
                self._qemu_stderr = None
        except Exception as e:
            logger.warning(f'Failed to close QEMU stderr: {e}')
        
        # Stop QEMU process if we started it
        if self.qemu_pid is not None:
            try:
                self.log('info', f'Stopping QEMU VM with PID: {self.qemu_pid}')
                os.kill(self.qemu_pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still alive
                try:
                    os.kill(self.qemu_pid, 0)
                    os.kill(self.qemu_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except Exception as e:
                self.log('warning', f'Failed to stop QEMU VM: {e}')
        
        # Call parent close
        super().close(rm_all_containers)
    
    @property
    def osworld_vm_url(self) -> str:
        """Get the OSWorld VM server URL."""
        return f'http://localhost:{self._vm_server_port}'
    
    @property
    def vnc_url(self) -> str:
        """Get the VNC URL for the VM."""
        return f'vnc://localhost:{self._vnc_port}'
    
    def get_vm_screenshot(self) -> bytes | None:
        """Get screenshot from the VM.
        
        Returns:
            PNG image bytes or None if failed
        """
        try:
            response = httpx.get(
                f'{self.osworld_vm_url}/screenshot',
                timeout=10.0
            )
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            self.log('error', f'Failed to get VM screenshot: {e}')
            return None
    
    def execute_vm_action(self, action_data: dict) -> dict:
        """Execute an action in the OSWorld VM.
        
        Args:
            action_data: Action data dictionary with 'action_type' and 'parameters'
            
        Returns:
            Response from OSWorld server
        """
        try:
            action_type = action_data.get('action_type')
            parameters = action_data.get('parameters', {})
            
            # Convert action to PyAutoGUI command
            pyautogui_command = self._action_to_pyautogui_command(action_type, parameters)
            
            if pyautogui_command is None:
                return {'status': 'error', 'message': f'Unknown action type: {action_type}'}
            
            # Execute the PyAutoGUI command
            command_list = ["python3", "-c", f"import pyautogui; import time; pyautogui.FAILSAFE = False; {pyautogui_command}"]
            payload = {"command": command_list, "shell": False}
            
            response = httpx.post(
                f'{self.osworld_vm_url}/execute',
                json=payload,
                timeout=30.0
            )
            return response.json()
        except Exception as e:
            self.log('error', f'Failed to execute VM action: {e}')
            return {'status': 'error', 'message': str(e)}
    
    def _action_to_pyautogui_command(self, action_type: str, parameters: dict) -> str | None:
        """Convert an action dictionary to a PyAutoGUI command string.
        
        Args:
            action_type: Type of action (e.g., 'CLICK', 'TYPING', 'PRESS')
            parameters: Action parameters
            
        Returns:
            PyAutoGUI command string or None if unknown action type
        """
        import random
        
        # For MOVE_TO actions with duration
        move_mode = random.choice([
            "pyautogui.easeInQuad", "pyautogui.easeOutQuad", "pyautogui.easeInOutQuad",
            "pyautogui.easeInBounce", "pyautogui.easeInElastic"
        ])
        duration = random.uniform(0.5, 1)
        
        if action_type == "CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            button = parameters.get('button', 'left')
            num_clicks = parameters.get('num_clicks', 1)
            
            if x is not None and y is not None:
                return f"pyautogui.click(x={x}, y={y}, button='{button}', clicks={num_clicks})"
            else:
                return "pyautogui.click()"
        
        elif action_type == "DOUBLE_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            if x is not None and y is not None:
                return f"pyautogui.doubleClick(x={x}, y={y})"
            else:
                return "pyautogui.doubleClick()"
        
        elif action_type == "RIGHT_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            if x is not None and y is not None:
                return f"pyautogui.rightClick(x={x}, y={y})"
            else:
                return "pyautogui.rightClick()"
        
        elif action_type == "MOVE_TO":
            x = parameters.get('x')
            y = parameters.get('y')
            if x is not None and y is not None:
                return f"pyautogui.moveTo({x}, {y}, {duration}, {move_mode})"
            else:
                return "pyautogui.moveTo()"
        
        elif action_type == "DRAG_TO":
            x = parameters.get('x')
            y = parameters.get('y')
            if x is not None and y is not None:
                return f"pyautogui.dragTo({x}, {y}, duration=1.0, button='left', mouseDownUp=True)"
            return None
        
        elif action_type == "SCROLL":
            dx = parameters.get('dx', 0)
            dy = parameters.get('dy', 0)
            commands = []
            if dx != 0:
                commands.append(f"pyautogui.hscroll({dx})")
            if dy != 0:
                commands.append(f"pyautogui.vscroll({dy})")
            return "; ".join(commands) if commands else None
        
        elif action_type == "TYPING":
            text = parameters.get('text', '')
            # Use repr() to properly escape the text
            return f"pyautogui.typewrite({repr(text)})"
        
        elif action_type == "PRESS":
            key = parameters.get('key', '')
            if isinstance(key, list):
                # Multiple keys - treat as hotkey
                keys_str = "', '".join(key)
                return f"pyautogui.hotkey('{keys_str}')"
            else:
                return f"pyautogui.press('{key}')"
        
        elif action_type == "HOTKEY":
            keys = parameters.get('keys', [])
            if isinstance(keys, list) and keys:
                keys_str = "', '".join(keys)
                return f"pyautogui.hotkey('{keys_str}')"
            return None
        
        elif action_type == "KEY_DOWN":
            key = parameters.get('key', '')
            return f"pyautogui.keyDown('{key}')"
        
        elif action_type == "KEY_UP":
            key = parameters.get('key', '')
            return f"pyautogui.keyUp('{key}')"
        
        elif action_type == "MOUSE_DOWN":
            button = parameters.get('button', 'left')
            return f"pyautogui.mouseDown(button='{button}')"
        
        elif action_type == "MOUSE_UP":
            button = parameters.get('button', 'left')
            return f"pyautogui.mouseUp(button='{button}')"
        
        else:
            return None

