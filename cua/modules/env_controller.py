from typing import Any, Dict

from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime


class EnvController:
    """
    Static Wrapper class that interfaces with OSWorldSingularityRuntime.
    """
    @staticmethod
    def initialize_runtime():
        """
        Initialize OSWorldSingularityRuntime.
        Used by DataCollector._init_worker to boot up the VM.
        """
        # todo
        pass

    @staticmethod
    def convert_som_action_to_runtime_action():
        """
        Convert model-generated action in Set-of-Marks context into OSWorld runtime action info
        """
        # todo return similar object as `action_info` in synthetic_data_generator.generate_action function
        # todo specifically, we need action_info Dict with "tool_name" and "params"
        # todo this includes converting selected Mark object into a solid coordinate in pixels
        pass

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



