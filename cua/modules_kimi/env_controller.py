import logging
import re
from typing import Dict, Tuple

from examples.setup import SetupController
from openhands.core.config import OpenHandsConfig
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.events.observation import ErrorObservation
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime
from openhands.storage import get_file_store
from openhands.core.logger import openhands_logger

logger = openhands_logger.getChild('kimi_env_controller')
logger.setLevel(logging.DEBUG)


class EnvController:
    """
    Static wrapper class that interfaces with OSWorldSingularityRuntime.
    Copied from modules/debug_env_controller.py for self-containment.
    """

    @staticmethod
    async def initialize_runtime(job_id: str, vm_image_path: str, os_type: str,
                                 osworld_setup: Dict) -> OSWorldSingularityRuntime:
        config = OpenHandsConfig()
        config.runtime = "osworld"
        config.sandbox.base_container_image = "ubuntu:24.04"
        config.sandbox.run_as_fakeroot = False
        config.sandbox.runtime_container_image = None

        file_store = get_file_store('local', f'/tmp/synthetic_data_gen_{job_id}')
        event_stream = EventStream(sid=job_id, file_store=file_store)

        logger.debug(f"[initialize_runtime] Creating runtime for {job_id}")
        logger.debug(f"[initialize_runtime]   VM image: {vm_image_path}")

        runtime = OSWorldSingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=job_id,
            os_type=os_type,
            vm_image_path=vm_image_path,
            attach_to_existing=False,
        )

        await runtime.connect()
        logger.debug(f"[initialize_runtime] Runtime initialized and connected for {job_id}")

        if osworld_setup and os_type == "linux" and osworld_setup.get('config'):
            logger.info(f"[initialize_runtime] [{job_id}] Setting up OSWorld with {len(osworld_setup['config'])} config step(s)")
            setup_controller = SetupController(
                vm_ip="127.0.0.1",
                server_port=runtime._vm_server_port,
                chromium_port=runtime._chromium_port,
                cache_dir="/tmp/osworld_example",
                client_password="password",
                runtime=runtime
            )
            try:
                await setup_controller.setup(osworld_setup['config'])
                logger.info(f"[initialize_runtime] [{job_id}] OSWorld setup completed successfully")
            except Exception as e:
                logger.error(f"[initialize_runtime] [{job_id}] OSWorld setup FAILED: {e}")
                logger.error(f"[initialize_runtime] [{job_id}] Failed config: {osworld_setup.get('config', [])}")
                raise
        else:
            logger.debug(f"[initialize_runtime] No OSWorld setup provided")

        return runtime

    @staticmethod
    def execute_pyautogui_command(runtime: OSWorldSingularityRuntime, pyautogui_command: str):
        pyautogui_action = OSWorldInteractiveAction(
            method="execute_python_command",
            params={"command": pyautogui_command},
        )
        result = runtime.run_action(pyautogui_action)

        if not isinstance(result, ErrorObservation):
            logger.debug("[execute_pyautogui_command] Action complete")
        else:
            logger.debug(f"[execute_pyautogui_command] Error in Action: {result}")

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
    def get_screenshot(runtime: OSWorldSingularityRuntime) -> bytes:
        screenshot = runtime.get_vm_screenshot()
        if not screenshot:
            logger.debug("Failed to get screenshot from runtime.")
            raise RuntimeError("Failed to get screenshot from runtime.")

        return screenshot
