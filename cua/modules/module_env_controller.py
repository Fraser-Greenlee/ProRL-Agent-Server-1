from typing import Any, Dict

from examples.setup import SetupController
from openhands.core.config import OpenHandsConfig
from openhands.events import EventStream
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime
from openhands.storage import get_file_store
from openhands.core.logger import openhands_logger


# NOTE: this is just a dummy class for now, plan to reuse them for parallel processing


# Create a child logger
logger = openhands_logger.getChild('env_controller')


class ModuleEnvController:
    """
    Static Wrapper class that interfaces with OSWorldSingularityRuntime.
    """
    @staticmethod
    async def initialize_runtime(job_id: str, vm_image_path: str, os_type: str,
                           osworld_setup: Dict) -> OSWorldSingularityRuntime:
        """
        Initialize OSWorldSingularityRuntime.
        Used by DataCollector._init_worker to boot up the VM.
        """
        config = OpenHandsConfig()
        config.runtime = "osworld"
        config.sandbox.base_container_image = "ubuntu:24.04"
        config.sandbox.run_as_fakeroot = False
        config.sandbox.runtime_container_image = None  # Trigger auto-build

        # Unique event stream per trajectory
        file_store = get_file_store('local', f'/tmp/synthetic_data_gen_{job_id}')
        event_stream = EventStream(sid=job_id, file_store=file_store)

        logger.info(f"[initialize_runtime] Creating runtime for {job_id}")
        logger.info(f"[initialize_runtime]   VM image: {vm_image_path}")
        logger.info(f"[initialize_runtime]   Base image: {config.sandbox.base_container_image}")

        runtime = OSWorldSingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=job_id,
            os_type=os_type,
            vm_image_path=vm_image_path,
            attach_to_existing=False,
        )

        logger.info(f"[initialize_runtime] Runtime object created, connecting to VM...")

        await runtime.connect()
        logger.info(f"[initialize_runtime] ✓ Runtime initialized and connected for {job_id}")
        logger.info(f"[initialize_runtime]   VM URL: {runtime.osworld_vm_url if hasattr(runtime, 'osworld_vm_url') else 'N/A'}")

        if osworld_setup and os_type == "linux":
            logger.info(f"[initialize_runtime] Setting up OSWorld...")
            setup_controller = SetupController(
                vm_ip="127.0.0.1",
                server_port=runtime._vm_server_port,
                chromium_port=runtime._chromium_port,
                cache_dir="/tmp/osworld_example",  # might need to be changed to a unique directory for each job
                client_password="password",
                runtime=runtime
            )
            await setup_controller.setup(osworld_setup['config'])
            logger.info(f"[initialize_runtime] ✓ OSWorld setup completed")
        else:
            logger.info(f"[initialize_runtime] No OSWorld setup provided")

        return runtime

    @staticmethod
    def execute_action(action_info: Dict[str, Any], runtime: OSWorldSingularityRuntime) -> Dict[str, Any]:
        """
        Execute the action selected by LLM.
        Updated for new schema.

        Args:
        action_info: Dict with "tool_name" and "params"
            "tool_name" should be one of TOOL_NAME_TO_ACTION_TYPE keys.
        runtime: Runtime instance to execute action on

        Returns:
        Observation: Dict with "success": bool, "exit code": 0 or -1
        """
        # todo reference: SyntheticDataGenerator.execute_action
        pass

    @staticmethod
    def save_screenshot(runtime: OSWorldSingularityRuntime, screenshot_path: str):
        """
        Save the current screenshot from the runtime.
        """
        # todo consider returning the base64 string of screenshot (because that's all we need for future processing)
        screenshot = runtime.get_vm_screenshot()
        if screenshot:
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            print(f"✓ Screenshot saved to {screenshot_path} | Size: {len(screenshot)} bytes")
        else:
            print("✗ Failed to get screenshot from runtime.")

        print()



