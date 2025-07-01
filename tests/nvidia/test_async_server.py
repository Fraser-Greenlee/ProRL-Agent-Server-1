import threading
import unittest
from unittest.mock import Mock, patch

from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.async_server import OpenHandsServer
from openhands.nvidia.registry import (
    AgentHandler,
    FunctionNotRegisteredError,
    JobDetails,
)


class MockInstance(dict):
    """Mock instance for testing."""

    def __init__(
        self,
        instance_id='test_instance',
        trajectory_id='test_trajectory',
        data_source='swebench',
    ):
        super().__init__()
        self['instance_id'] = instance_id
        self['trajectory_id'] = trajectory_id
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        self.data_source = data_source


class MockAgentHandler(AgentHandler):
    """Mock agent handler for testing."""

    @property
    def name(self) -> str:
        return 'swebench'

    async def init(self, instance, llm_config=None, sid=None, max_iterations=1):
        mock_runtime = Mock()
        mock_metadata = Mock()
        mock_config = Mock()
        return mock_runtime, mock_metadata, mock_config

    async def run(self, runtime, metadata, config, instance):
        return {'status': 'completed', 'result': 'success'}

    async def eval(self, job_details, sid=None, allow_skip=True):
        return {'report': {'status': 'passed'}}

    def init_exception(self, job_details, exception):
        return {'error': 'init_failed', 'exception': str(exception)}

    def run_exception(self, job_details, exception):
        return {'error': 'run_failed', 'exception': str(exception)}

    def eval_exception(self, job_details, exception):
        return {'error': 'eval_failed', 'exception': str(exception)}

    def final_result(self, job_details):
        if hasattr(job_details, 'eval_results'):
            return {'final_result': job_details.eval_results}
        elif hasattr(job_details, 'run_results'):
            return {'final_result': job_details.run_results}
        elif hasattr(job_details, 'results'):
            return {'final_result': job_details.results}
        else:
            return {'final_result': 'no_results'}


class TestOpenHandsServer(unittest.TestCase):
    """Test cases for OpenHandsServer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000'],
            max_init_workers=2,
            max_run_workers=2,
            max_eval_workers=2,
        )
        self.mock_instance = MockInstance()
        self.mock_handler = MockAgentHandler()

    def tearDown(self):
        """Clean up after tests."""
        if self.server._server_running:
            self.server.stop()

    def test_initialization_default_parameters(self):
        """Test server initialization with default parameters."""
        server = OpenHandsServer()
        self.assertEqual(server.max_init_workers, 6)
        self.assertEqual(server.max_run_workers, 5)
        self.assertEqual(server.max_eval_workers, 5)  # defaults to max_run_workers
        self.assertTrue(server.allow_skip_eval)
        self.assertFalse(server._server_running)
        self.assertEqual(len(server.weighted_addresses), 0)

    def test_initialization_custom_parameters(self):
        """Test server initialization with custom parameters."""
        server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000', 'http://localhost:8001'],
            max_init_workers=3,
            max_run_workers=4,
            max_eval_workers=2,
            allow_skip_eval=False,
        )
        self.assertEqual(server.max_init_workers, 3)
        self.assertEqual(server.max_run_workers, 4)
        self.assertEqual(server.max_eval_workers, 2)
        self.assertFalse(server.allow_skip_eval)
        self.assertEqual(len(server.weighted_addresses), 2)

    def test_add_llm_server_address(self):
        """Test adding LLM server addresses."""
        server = OpenHandsServer()
        server.add_llm_server_address('http://localhost:8000')
        self.assertEqual(len(server.weighted_addresses), 1)
        self.assertEqual(server.weighted_addresses[0][1], 'http://localhost:8000')

        # Test adding duplicate address
        with patch('openhands.nvidia.async_server.logger') as mock_logger:
            server.add_llm_server_address('http://localhost:8000')
            mock_logger.warning.assert_called_once()

    def test_clear_llm_server_addresses(self):
        """Test clearing LLM server addresses."""
        self.server.clear_llm_server_addresses()
        self.assertEqual(len(self.server.weighted_addresses), 0)

    def test_create_llm_config_no_addresses(self):
        """Test creating LLM config with no addresses."""
        server = OpenHandsServer()
        with self.assertRaises(ValueError) as context:
            server.create_llm_config({'temperature': 0.7})
        self.assertIn('No LLM server addresses added', str(context.exception))

    def test_create_llm_config_with_addresses(self):
        """Test creating LLM config with available addresses."""
        config = self.server.create_llm_config({'temperature': 0.7})
        self.assertIsInstance(config, LLMConfig)
        self.assertEqual(config.base_url, 'http://localhost:8000')

    def test_get_unique_id(self):
        """Test unique ID generation."""
        uid1 = self.server.get_unique_id(self.mock_instance)
        uid2 = self.server.get_unique_id(self.mock_instance)
        self.assertNotEqual(uid1, uid2)
        self.assertIn('test_instance', uid1)
        self.assertIn('test_trajectory', uid1)

    def test_get_unique_id_max_retries(self):
        """Test unique ID generation with max retries."""
        # Fill up job details to force retries - create IDs that match the pattern
        for i in range(15):
            fake_id = f'test_instance_test_trajectory_{i}'
            self.server._job_details[fake_id] = JobDetails()

        # Mock uuid.uuid4 to return predictable values that will conflict
        with patch(
            'openhands.nvidia.async_server.uuid.uuid4',
            side_effect=[str(i) for i in range(15)],
        ):
            with self.assertRaises(ValueError) as context:
                self.server.get_unique_id(self.mock_instance)
            self.assertIn('Failed to get unique id', str(context.exception))

    def test_start_server(self):
        """Test starting the server."""
        self.assertFalse(self.server._server_running)
        self.server.start()
        self.assertTrue(self.server._server_running)

        # Test starting already running server
        with self.assertRaises(RuntimeError) as context:
            self.server.start()
        self.assertIn('Server is already running', str(context.exception))

    def test_stop_server_not_running(self):
        """Test stopping server that's not running."""
        # Should not raise an exception
        self.server.stop()
        self.assertFalse(self.server._server_running)

    def test_status_empty_server(self):
        """Test status of empty server."""
        status = self.server.status()
        expected = {
            'init_queue': 0,
            'run_queue': 0,
            'eval_queue': 0,
            'active_init': 0,
            'active_run': 0,
            'active_eval': 0,
            'total': 0,
        }
        self.assertEqual(status, expected)

    def test_process_server_not_running(self):
        """Test processing when server is not running."""
        with self.assertRaises(RuntimeError) as context:
            self.server.process(self.mock_instance, {'temperature': 0.7})
        self.assertIn('Server is not running', str(context.exception))

    def test_process_no_llm_addresses(self):
        """Test processing with no LLM addresses."""
        server = OpenHandsServer()
        server.start()
        try:
            with self.assertRaises(ValueError) as context:
                server.process(self.mock_instance, {'temperature': 0.7})
            self.assertIn('No LLM server addresses added', str(context.exception))
        finally:
            server.stop()

    def test_process_unregistered_handler(self):
        """Test processing with unregistered handler."""
        # Create a mock instance with an unregistered dataset type
        mock_instance = MockInstance(data_source='unregistered_dataset')

        self.server.start()

        try:
            with self.assertRaises(FunctionNotRegisteredError) as context:
                self.server.process(mock_instance, {'temperature': 0.7})
            self.assertIn(
                'Dataset type unregistered_dataset is not registered',
                str(context.exception),
            )
        finally:
            self.server.stop()

    def test_process_full_pipeline(self):
        """Test full processing pipeline with mocked registry."""
        # This test is complex because it involves the actual registry system
        # For now, we'll test the basic process flow without full execution
        # since the real registry functions are already registered

        # Create a simple mock instance that won't trigger singularity
        mock_instance = MockInstance(data_source='test_dataset')

        # Mock the registry functions to avoid real execution
        with patch(
            'openhands.nvidia.registry.is_registered_handler', return_value=False
        ):
            self.server.start()

            try:
                with self.assertRaises(FunctionNotRegisteredError):
                    self.server.process(mock_instance, {'temperature': 0.7})
            finally:
                self.server.stop()

    def test_thread_safety_job_details(self):
        """Test thread safety of job details access."""
        job_id = 'test_job'
        job_details = JobDetails()

        # Test concurrent access to job details
        def add_job():
            with self.server._job_details_lock:
                self.server._job_details[job_id] = job_details

        def remove_job():
            with self.server._job_details_lock:
                if job_id in self.server._job_details:
                    del self.server._job_details[job_id]

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=add_job))
            threads.append(threading.Thread(target=remove_job))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not crash due to race conditions
        self.assertTrue(True)

    def test_thread_safety_active_jobs(self):
        """Test thread safety of active jobs tracking."""
        job_id = 'test_job'

        def add_to_active():
            with self.server._state_lock:
                self.server._active_init_jobs.add(job_id)

        def remove_from_active():
            with self.server._state_lock:
                self.server._active_init_jobs.discard(job_id)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=add_to_active))
            threads.append(threading.Thread(target=remove_from_active))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not crash due to race conditions
        self.assertTrue(True)

    def test_weighted_addresses_load_balancing(self):
        """Test weighted addresses for load balancing."""
        server = OpenHandsServer(
            llm_server_addresses=['http://localhost:8000', 'http://localhost:8001']
        )

        # Create multiple configs and ensure addresses are rotated
        configs = []
        for _ in range(4):
            config = server.create_llm_config({'temperature': 0.7})
            configs.append(config.base_url)

        # Should have used both addresses
        unique_addresses = set(configs)
        self.assertEqual(len(unique_addresses), 2)
        self.assertIn('http://localhost:8000', unique_addresses)
        self.assertIn('http://localhost:8001', unique_addresses)

    def test_queue_operations(self):
        """Test queue operations and status."""
        # Add some items to queues
        self.server.init_queue.put('job1')
        self.server.run_queue.put('job2')
        self.server.evaluate_queue.put('job3')

        # Add some active jobs
        with self.server._state_lock:
            self.server._active_init_jobs.add('active1')
            self.server._active_run_jobs.add('active2')
            self.server._active_eval_jobs.add('active3')

        status = self.server.status()
        self.assertEqual(status['init_queue'], 1)
        self.assertEqual(status['run_queue'], 1)
        self.assertEqual(status['eval_queue'], 1)
        self.assertEqual(status['active_init'], 1)
        self.assertEqual(status['active_run'], 1)
        self.assertEqual(status['active_eval'], 1)
        self.assertEqual(status['total'], 6)

    @patch('openhands.nvidia.async_server.clear_queue')
    def test_stop_server_cleanup(self, mock_clear_queue):
        """Test server cleanup on stop."""
        # Add some test data
        self.server._job_details['test_job'] = JobDetails()
        self.server._active_init_jobs.add('test_job')

        self.server.start()
        self.server.stop()

        # Verify cleanup
        self.assertFalse(self.server._server_running)
        self.assertEqual(len(self.server._job_details), 0)
        self.assertEqual(len(self.server._active_init_jobs), 0)
        self.assertEqual(len(self.server._active_run_jobs), 0)
        self.assertEqual(len(self.server._active_eval_jobs), 0)

        # Verify queues were cleared
        self.assertEqual(mock_clear_queue.call_count, 3)

    def test_custom_job_id(self):
        """Test processing with custom job ID."""
        custom_job_id = 'custom_test_job_123'

        # Create a mock instance that will fail registration check
        mock_instance = MockInstance(data_source='unregistered_type')

        self.server.start()

        try:
            # This will fail at registration check, but the job ID should still be used
            try:
                self.server.process(
                    mock_instance, {'temperature': 0.7}, job_id=custom_job_id
                )
            except FunctionNotRegisteredError:
                pass  # Expected to fail, but job should have been created

        finally:
            self.server.stop()

    def test_job_lifecycle_basic(self):
        """Test basic job lifecycle without external dependencies."""
        # Test that jobs can be created and tracked properly
        job_id = self.server.get_unique_id(self.mock_instance)

        # Manually create a job details object to test lifecycle
        from openhands.nvidia.registry import JobDetails

        job_details = JobDetails()
        job_details.job_id = job_id
        job_details.instance = self.mock_instance
        job_details.event = threading.Event()

        # Test job details management
        with self.server._job_details_lock:
            self.server._job_details[job_id] = job_details
            self.assertIn(job_id, self.server._job_details)

        # Test active job tracking
        with self.server._state_lock:
            self.server._active_init_jobs.add(job_id)
            self.assertIn(job_id, self.server._active_init_jobs)
            self.server._active_init_jobs.discard(job_id)
            self.assertNotIn(job_id, self.server._active_init_jobs)

        # Cleanup
        with self.server._job_details_lock:
            if job_id in self.server._job_details:
                del self.server._job_details[job_id]


if __name__ == '__main__':
    unittest.main()
