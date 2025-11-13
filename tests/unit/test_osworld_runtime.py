"""Unit tests for OSWorld Singularity Runtime."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from openhands.core.config import OpenHandsConfig
from openhands.events import EventStream
from openhands.runtime.impl.singularity.osworld_singularity_runtime import (
    OSWorldSingularityRuntime,
    OSWORLD_VM_SERVER_PORT_RANGE,
    OSWORLD_VNC_PORT_RANGE,
)


class TestOSWorldSingularityRuntime(unittest.TestCase):
    """Test cases for OSWorld Singularity Runtime."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OpenHandsConfig()
        self.config.runtime = 'osworld'
        self.config.sandbox.base_container_image = 'ubuntu:24.04'
        
        # Mock event stream
        self.event_stream = Mock(spec=EventStream)
        self.event_stream.file_store = Mock()
        self.event_stream.file_store.write = Mock()
        self.event_stream.file_store.read = Mock(side_effect=FileNotFoundError)
        self.event_stream.file_store.delete = Mock()

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            self.assertEqual(runtime.os_type, 'linux')
            self.assertEqual(runtime._vm_server_port, -1)
            self.assertEqual(runtime._vnc_port, -1)
            self.assertIsNone(runtime.qemu_process)

    def test_init_with_windows_os(self):
        """Test initialization with Windows OS type."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                os_type='windows',
                attach_to_existing=False,
            )
            
            self.assertEqual(runtime.os_type, 'windows')
            self.assertIn('Windows-10-x64.qcow2', runtime.vm_image_path)

    def test_get_default_vm_image_path_linux(self):
        """Test getting default VM image path for Linux."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                os_type='linux',
                attach_to_existing=False,
            )
            
            path = runtime._get_default_vm_image_path()
            self.assertIn('Ubuntu.qcow2', path)

    def test_get_default_vm_image_path_windows(self):
        """Test getting default VM image path for Windows."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                os_type='windows',
                attach_to_existing=False,
            )
            
            path = runtime._get_default_vm_image_path()
            self.assertIn('Windows-10-x64.qcow2', path)

    def test_allocate_osworld_ports(self):
        """Test port allocation for OSWorld services."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            with patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.find_available_tcp_port') as mock_find_port:
                mock_find_port.side_effect = [15000, 18000]
                
                vm_port, vnc_port = runtime._allocate_osworld_ports()
                
                self.assertEqual(vm_port, 15000)
                self.assertEqual(vnc_port, 18000)
                self.assertEqual(mock_find_port.call_count, 2)

    def test_get_qemu_command_linux(self):
        """Test QEMU command generation for Linux VM."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            with tempfile.NamedTemporaryFile(suffix='.qcow2', delete=False) as tmp:
                vm_image_path = tmp.name
                
            try:
                runtime = OSWorldSingularityRuntime(
                    config=self.config,
                    event_stream=self.event_stream,
                    sid='test-session',
                    os_type='linux',
                    vm_image_path=vm_image_path,
                    attach_to_existing=False,
                )
                
                runtime._vm_server_port = 15000
                runtime._vnc_port = 18000
                
                cmd = runtime._get_qemu_command()
                
                self.assertIn('qemu-system-x86_64', cmd)
                self.assertIn('-enable-kvm', cmd)
                self.assertIn('-m', cmd)
                self.assertIn('16G', cmd)
                self.assertIn(f'file={vm_image_path}', ' '.join(cmd))
                self.assertIn('hostfwd=tcp::15000-:5000', ' '.join(cmd))
                self.assertIn('hostfwd=tcp::18000-:8006', ' '.join(cmd))
            finally:
                if os.path.exists(vm_image_path):
                    os.unlink(vm_image_path)

    def test_osworld_vm_url(self):
        """Test OSWorld VM URL property."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vm_server_port = 15000
            
            url = runtime.osworld_vm_url
            self.assertEqual(url, 'http://localhost:15000')

    def test_vnc_url(self):
        """Test VNC URL property."""
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vnc_port = 18000
            
            url = runtime.vnc_url
            self.assertEqual(url, 'vnc://localhost:18000')

    @patch('httpx.get')
    def test_check_if_alive_success(self, mock_get):
        """Test checking if VM is alive - success case."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vm_server_port = 15000
            
            # Should not raise exception
            runtime.check_if_alive()
            
            mock_get.assert_called_once_with(
                'http://localhost:15000/screenshot',
                timeout=5.0
            )

    @patch('httpx.get')
    def test_check_if_alive_failure(self, mock_get):
        """Test checking if VM is alive - failure case."""
        mock_get.side_effect = Exception('Connection refused')
        
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vm_server_port = 15000
            
            from openhands.core.exceptions import AgentRuntimeDisconnectedError
            with self.assertRaises(AgentRuntimeDisconnectedError):
                runtime.check_if_alive()

    @patch('httpx.get')
    def test_get_vm_screenshot(self, mock_get):
        """Test getting VM screenshot."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'fake-png-data'
        mock_get.return_value = mock_response
        
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vm_server_port = 15000
            
            screenshot = runtime.get_vm_screenshot()
            
            self.assertEqual(screenshot, b'fake-png-data')
            mock_get.assert_called_once()

    @patch('httpx.post')
    def test_execute_vm_action(self, mock_post):
        """Test executing VM action."""
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success', 'result': 'action executed'}
        mock_post.return_value = mock_response
        
        with patch.object(
            OSWorldSingularityRuntime, '_check_singularity_availability'
        ), patch('openhands.runtime.impl.singularity.osworld_singularity_runtime.allocate_loopback_ip'):
            runtime = OSWorldSingularityRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid='test-session',
                attach_to_existing=False,
            )
            
            runtime._vm_server_port = 15000
            
            action_data = {
                'action_type': 'CLICK',
                'parameters': {'x': 100, 'y': 200}
            }
            
            result = runtime.execute_vm_action(action_data)
            
            self.assertEqual(result['status'], 'success')
            mock_post.assert_called_once_with(
                'http://localhost:15000/execute',
                json=action_data,
                timeout=30.0
            )


if __name__ == '__main__':
    unittest.main()

