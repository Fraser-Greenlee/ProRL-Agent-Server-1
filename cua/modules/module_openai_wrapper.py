from argparse import Namespace
from typing import Dict, List
from PIL import Image

from openai import OpenAI
from openhands.core.logger import openhands_logger

# Create a child logger
logger = openhands_logger.getChild('openai_wrapper')


class OpenAIWrapper:
    """
    Wrapper class that supports all interactions with remote vLLM server.
    """
    def __init__(self, args: Namespace):
        self.client = OpenAI(
            base_url=f"http://{args.explorer_node}:8000/v1",
            api_key="gen",
        )
        self.model_name = args.model_name

        if args.min_pixels != -1 and args.max_pixels != -1:
            self.mm_processor_kwargs = {
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        else:
            self.mm_processor_kwargs = None

    def prompt_vlm(self, messages: List, n: int = 1, temperature: float = 0.7, max_tokens: int = 8192) -> List[str]:
        """
        Helper function to prompt VLM with messages.
        Use with try...except phrase to process timeout error.
        """
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={
                "mm_processor_kwargs": None,
            },
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=int(600),
        )

        return [choice.message.content for choice in chat_response.choices]

    # todo build helper function to make summary of observation given the entire trajectory so far

    def prompt_action(self):
        # todo for now, we consider atomic, single function based action.
        # todo need to filter out wrong generations that does not abide by the function schema (i.e., json format)
        # todo by using parse_action_from_generation.
        # todo if generation does not abide by function schema, re-generate up to max_retry_for_action_generation
        pass

    def parse_action_from_generation(self):
        # todo parse action (i.e., action schema) from the generation
        # the parsed action should abide by the SOM action scheme
        # (EnvController will convert this to executable OSWorld action)

        # todo return similar object as `action_info` in synthetic_data_generator.generate_action function
        # todo specifically, we need action_info Dict with "tool_name" and "params"
        # todo this includes converting selected Mark object into a solid coordinate in pixels
        pass

    def prompt_goal(self):
        # todo similar to SyntheticDataGenerator.generate_goal, generate high-level sub-goal to pick random actions
        pass

    def prompt_goal_with_persona(self, persona: Dict):
        # todo similar to SyntheticDataGenerator.generate_goal, generate high-level sub-goal to pick random actions
        # todo conditioned on persona
        pass

    @staticmethod
    def prepare_generate_goal_messages(current_screenshot: Image.Image, persona: Dict, previous_goals: List[str]):
        # todo
        # You are an AI agent exploring a Ubuntu desktop environment.
        #
        # Your task is to imagine ONE reasonable sub-goal you could achieve based on the current screen state.
        #
        # Respond with a single, specific goal.

        # These are the previous goals you pursued:
        # Some of these goals may not have been fully achieved. Based on the current screen state, determine whether
        # your last goal has been achieved, and try to make a natural connection when coming up with a new goal.
        # (e.g. if the current state is to go to a different website and the screenshot tells you that you just
        # clicked on the URL field on chrome, your next goal could be typing the URL.

        # Guidelines
        # - Don't ask clarification questions - just generate a simple goal
        # - A larger objective is irrelevant. We just need a local sub-goal to navigate the system. Don't ask for it!
        # - Choose realistic, achievable goals from visible UI elements
        # - Goals should be specific and actionable (e.g., "Type 'news' in search box", "Open Google Chrome")
        # - Goals must be atomic - ONE action at a time. Click is one goal, type is another goal.
        # - Be curious and explore different parts of the system
        # - If the same goal is generated multiple times, it means you might have been stuck in a loop.
        # Try to backtrack and generate a different goal.
        # - Generate coherent goal sequences (e.g., if browser is open, search for something)
        # - Don't do random app switches in your goals
        # - If you cannot find some file in your previous goals, most likely it doesn't exist and hallucinated.
        # Do not try to find it again, and try to generate a different goal.
        # - Goals must be achievable with current screen state.

        pass









