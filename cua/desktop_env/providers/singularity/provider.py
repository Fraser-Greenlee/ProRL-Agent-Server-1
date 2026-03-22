import logging
import os
import random
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.singularity.SingularityProvider")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

WAIT_TIME = 3
RETRY_INTERVAL = 5

DEFAULT_SIF_PATH = "/root/singularity_images/osworld_singularity_ubuntu_24_04.sif"

# Port ranges for OSWorld VM services (matches ProRL-Agent-Server)
OSWORLD_VM_SERVER_PORT_RANGE = (15000, 19999)  # OSWorld Flask server inside VM (port 5000)
OSWORLD_VNC_PORT_RANGE = (18000, 22999)        # VNC server for VM display
OSWORLD_CHROMIUM_PORT_RANGE = (19000, 22999)   # Chrome DevTools Protocol (port 9222)
OSWORLD_VLC_PORT_RANGE = (20000, 22999)        # VLC web interface (port 8080)


class PortAllocationError(Exception):
    pass


class SingularityProvider(Provider):
    _port_allocation_lock = threading.Lock()
    
    def __init__(self, region: str = None):
        super().__init__(region)
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.qemu_process: subprocess.Popen = None
        self.qemu_pid: int = None
        self._qemu_stdout = None
        self._qemu_stderr = None
        
        self.environment = {
            "DISK_SIZE": "32G",
            "RAM_SIZE": "8G",
            "CPU_CORES": "4"
        }
        
        self._check_singularity_availability()

    @staticmethod
    def _check_singularity_availability():
        """Check if Singularity/Apptainer is available on the system."""
        try:
            result = subprocess.run(
                ['singularity', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f'Singularity version: {result.stdout.strip()}')
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                'Singularity is not available on this system. '
                'Please install Singularity/Apptainer to use SingularityProvider.'
            ) from e

    @staticmethod
    def _check_port_available(port: int) -> bool:
        """Check if a port is available by actually trying to bind to it."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _find_available_port(self, min_port: int, max_port: int, max_attempts: int = 50) -> int:
        """Find an available TCP port in a specified range.
        
        Uses random selection and socket binding for reliable port allocation.
        """
        rng = random.SystemRandom()
        ports = list(range(min_port, max_port + 1))
        rng.shuffle(ports)
        
        for port in ports[:max_attempts]:
            if self._check_port_available(port):
                return port
        raise PortAllocationError(f"No available ports found in range {min_port}-{max_port}")

    def _allocate_ports(self) -> tuple[int, int, int, int]:
        """Allocate ports for all OSWorld VM services.
        
        Returns:
            Tuple of (server_port, vnc_port, chromium_port, vlc_port)
        """
        with SingularityProvider._port_allocation_lock:
            server_port = self._find_available_port(
                OSWORLD_VM_SERVER_PORT_RANGE[0],
                OSWORLD_VM_SERVER_PORT_RANGE[1]
            )
            vnc_port = self._find_available_port(
                OSWORLD_VNC_PORT_RANGE[0],
                OSWORLD_VNC_PORT_RANGE[1]
            )
            chromium_port = self._find_available_port(
                OSWORLD_CHROMIUM_PORT_RANGE[0],
                OSWORLD_CHROMIUM_PORT_RANGE[1]
            )
            vlc_port = self._find_available_port(
                OSWORLD_VLC_PORT_RANGE[0],
                OSWORLD_VLC_PORT_RANGE[1]
            )
            return server_port, vnc_port, chromium_port, vlc_port

    def _check_kvm_available(self) -> bool:
        """Check if KVM is available and accessible.
        
        Set DISABLE_KVM=1 environment variable to force disable KVM
        (useful for WSL2 or environments where KVM exists but can't be used).
        """
        if os.environ.get('DISABLE_KVM', '').lower() in ('1', 'true', 'yes'):
            logger.info('KVM disabled via DISABLE_KVM environment variable')
            return False
        
        kvm_device = '/dev/kvm'
        if not os.path.exists(kvm_device):
            logger.debug(f'KVM device {kvm_device} does not exist')
            return False
        if not os.access(kvm_device, os.R_OK | os.W_OK):
            logger.debug(f'KVM device {kvm_device} exists but lacks read/write permissions')
            return False
        logger.debug(f'KVM device {kvm_device} is available and accessible')
        return True

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def _get_qemu_command(self, vm_image_path: str) -> list[str]:
        """Build QEMU command based on configuration."""
        vm_image_filename = os.path.basename(vm_image_path)
        vm_image_container_path = f'/OS_images/{vm_image_filename}'
        
        kvm_available = self._check_kvm_available()
        if kvm_available:
            logger.info('KVM is available, enabling hardware acceleration')
        else:
            logger.warning('KVM is not available, running QEMU in emulation mode (slower)')
        
        cmd = [
            'qemu-system-x86_64',
            '-bios', '/usr/share/ovmf/OVMF.fd',
            '-machine', 'q35',
        ]
        
        if kvm_available:
            cmd.extend(['-cpu', 'host', '-enable-kvm'])
        else:
            cmd.extend(['-cpu', 'qemu64'])
        
        portfwd = (
            f'user,id=net0,'
            f'hostfwd=tcp::{self.server_port}-:5000,'
            f'hostfwd=tcp::{self.chromium_port}-:9222,'
            f'hostfwd=tcp::{self.vlc_port}-:8080'
        )
        
        vnc_display = self.vnc_port - 5900
        
        cmd.extend([
            '-m', '8G',
            '-smp', '4',
            '-drive', f'file={vm_image_container_path},if=ide',
            '-netdev', portfwd,
            '-device', 'virtio-net-pci,netdev=net0',
            '-snapshot',
            '-vnc', f'0.0.0.0:{vnc_display}',
        ])
        
        return cmd

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        """Start QEMU VM inside Singularity container."""
        sif_path = os.environ.get("SINGULARITY_SIF_PATH", DEFAULT_SIF_PATH)
        if not os.path.exists(sif_path):
            raise FileNotFoundError(
                f"Singularity image not found: {sif_path}. "
                f"Please build or download it first. See SINGULARITY_GUIDELINE.md"
            )
        
        if not os.path.exists(path_to_vm):
            raise FileNotFoundError(
                f"VM image not found: {path_to_vm}. "
                f"Please download the QCOW2 image first."
            )
        
        try:
            self.server_port, self.vnc_port, self.chromium_port, self.vlc_port = self._allocate_ports()

            logger.info(
                f"Allocated ports - VNC: {self.vnc_port}, "
                f"Server: {self.server_port}, Chrome: {self.chromium_port}, VLC: {self.vlc_port}"
            )
            
            vm_image_dir = os.path.dirname(os.path.abspath(path_to_vm))
            
            env_vars = {
                'VM_SERVER_PORT': str(self.server_port),
                'VNC_PORT': str(self.vnc_port),
                'CHROMIUM_PORT': str(self.chromium_port),
                'VLC_PORT': str(self.vlc_port),
                'OS_TYPE': os_type,
            }
            
            cmd = [
                'singularity', 'exec',
                '--pid',
                '--writable-tmpfs',
                '--no-mount', 'cwd,tmp',
                '--home', '/root',
            ]
            
            for key, value in env_vars.items():
                cmd.extend(['--env', f'{key}={value}'])
            
            cmd.extend(['--bind', f'{vm_image_dir}:/OS_images'])
            
            cmd.append(sif_path)
            
            qemu_cmd = self._get_qemu_command(path_to_vm)
            cmd.extend(qemu_cmd)
            
            logger.info(f"Starting QEMU VM with command: {' '.join(cmd)}")
            
            log_dir = Path("/tmp/osworld_singularity_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            qemu_stdout_path = log_dir / f"qemu_{timestamp}.out"
            qemu_stderr_path = log_dir / f"qemu_{timestamp}.err"
            
            self._qemu_stdout = open(qemu_stdout_path, 'w')
            self._qemu_stderr = open(qemu_stderr_path, 'w')
            
            self.qemu_process = subprocess.Popen(
                cmd,
                stdout=self._qemu_stdout,
                stderr=self._qemu_stderr,
                text=True,
                start_new_session=True
            )
            
            self.qemu_pid = self.qemu_process.pid
            
            time.sleep(2)
            if self.qemu_process.poll() is not None:
                self._qemu_stdout.close()
                self._qemu_stderr.close()
                with open(qemu_stderr_path, 'r') as f:
                    error_output = f.read()
                
                # Check for KVM permission errors and provide helpful message
                if 'kvm' in error_output.lower() and 'permission denied' in error_output.lower():
                    raise RuntimeError(
                        f'QEMU failed to start due to KVM permission issues.\n'
                        f'Error: {error_output}\n\n'
                        f'To fix this, either:\n'
                        f'1. Run with KVM disabled: DISABLE_KVM=1 python quickstart.py ...\n'
                        f'2. Add your user to the kvm group: sudo usermod -aG kvm $USER\n'
                        f'3. In WSL2, KVM may not be available - use DISABLE_KVM=1'
                    )
                
                raise RuntimeError(
                    f'QEMU failed to start. Return code: {self.qemu_process.returncode}\n'
                    f'Error: {error_output}'
                )
            
            logger.info(f"QEMU VM started with PID: {self.qemu_pid}")
            logger.info(f"QEMU logs: stdout={qemu_stdout_path}, stderr={qemu_stderr_path}")
            
            self._wait_for_vm_ready()
            
            logger.info("VM is ready!")
            logger.info(f"  OSWorld API: http://localhost:{self.server_port}")
            logger.info(f"  VNC: vnc://localhost:{self.vnc_port} (display :{self.vnc_port - 5900})")
            logger.info(f"  Chrome DevTools: http://localhost:{self.chromium_port}")
            logger.info(f"  VLC Web Interface: http://localhost:{self.vlc_port}")

        except Exception as e:
            self._cleanup_process()
            raise e

    def _cleanup_process(self):
        """Clean up QEMU process and file handles."""
        if self._qemu_stdout:
            try:
                self._qemu_stdout.close()
            except Exception:
                pass
            self._qemu_stdout = None
            
        if self._qemu_stderr:
            try:
                self._qemu_stderr.close()
            except Exception:
                pass
            self._qemu_stderr = None
        
        if self.qemu_pid is not None:
            try:
                os.kill(self.qemu_pid, signal.SIGTERM)
                time.sleep(2)
                try:
                    os.kill(self.qemu_pid, 0)
                    os.kill(self.qemu_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except Exception as e:
                logger.warning(f"Failed to kill QEMU process: {e}")
            self.qemu_pid = None
        
        self.qemu_process = None

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Singularity provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        logger.info("Stopping VM...")
        self._cleanup_process()
        
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        
        time.sleep(WAIT_TIME)
        logger.info("VM stopped.")
