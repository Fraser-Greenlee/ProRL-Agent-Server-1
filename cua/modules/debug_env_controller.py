import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import ipdb

from examples.setup import SetupController
from openhands.core.config import OpenHandsConfig
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime
from openhands.storage import get_file_store
from openhands.core.logger import openhands_logger

# Create a child logger
logger = openhands_logger.getChild('env_controller')
logger.setLevel(logging.DEBUG)


class EnvController:
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

        logger.debug(f"[initialize_runtime] Creating runtime for {job_id}")
        logger.debug(f"[initialize_runtime]   VM image: {vm_image_path}")
        logger.debug(f"[initialize_runtime]   Base image: {config.sandbox.base_container_image}")

        runtime = OSWorldSingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=job_id,
            os_type=os_type,
            vm_image_path=vm_image_path,
            attach_to_existing=False,
        )

        logger.debug(f"[initialize_runtime] Runtime object created, connecting to VM...")

        await runtime.connect()
        logger.debug(f"[initialize_runtime] ✓ Runtime initialized and connected for {job_id}")
        logger.debug(f"[initialize_runtime]   VM URL: {runtime.osworld_vm_url if hasattr(runtime, 'osworld_vm_url') else 'N/A'}")

        if osworld_setup and os_type == "linux":
            logger.debug(f"[initialize_runtime] Setting up OSWorld...")
            setup_controller = SetupController(
                vm_ip="127.0.0.1",
                server_port=runtime._vm_server_port,
                chromium_port=runtime._chromium_port,
                cache_dir="/tmp/osworld_example",  # might need to be changed to a unique directory for each job
                client_password="password",
                runtime=runtime
            )
            await setup_controller.setup(osworld_setup['config'])
            logger.debug(f"[initialize_runtime] ✓ OSWorld setup completed")
        else:
            logger.debug(f"[initialize_runtime] No OSWorld setup provided")

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
    def get_screen_size(runtime: OSWorldSingularityRuntime) -> Tuple[int, int]:
        observation = runtime.run_action(OSWorldInteractiveAction(
            method="get_vm_screen_size",
            params={},
            thought=""
        ))

        assert hasattr(observation, "content"), "get_screen_size failed."

        match = re.search(r"Width: (\d+), Height: (\d+)", observation.content)
        width, height = int(match.group(1)), int(match.group(2))

        return width, height

    @staticmethod
    def get_and_save_screenshot(runtime: OSWorldSingularityRuntime, screenshot_path: Path = None) -> bytes:
        """
        Returns the current screenshot from the runtime, in base64 format.
        If screenshot_path is set, save the screenshot as png.
        """
        screenshot = runtime.get_vm_screenshot()
        if not screenshot:
            logger.debug("✗ Failed to get screenshot from runtime.")
            raise RuntimeError("Failed to get screenshot from runtime.")

        if screenshot_path:
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            logger.debug(f"✓ Screenshot saved to {str(screenshot_path)} | Size: {len(screenshot)} bytes")

        return screenshot



