from typing import Dict

from openai import OpenAI


class OpenAIWrapper:
    """
    Wrapper class that supports all interactions with remote vLLM server.
    """
    def __init__(self):
        # todo
        pass

    # todo build helper function to make summary of observation given the entire trajectory so far

    def prompt_action(self):
        # todo for now, we consider atomic, single function based action.
        # todo need to filter out wrong generations that does not abide by the function syntax (i.e., json format)
        # todo by using parse_action_from_generation.
        pass

    def parse_action_from_generation(self):
        # todo parse action (i.e., action schema) from the generation
        # the parsed action should abide by the SOM action scheme
        # (EnvController will convert this to executable OSWorld action)
        pass

    def prompt_goal(self):
        # todo similar to SyntheticDataGenerator.generate_goal, generate high-level sub-goal to pick random actions
        pass

    def prompt_goal_with_persona(self, persona: Dict):
        # todo similar to SyntheticDataGenerator.generate_goal, generate high-level sub-goal to pick random actions
        # todo conditioned on persona
        pass






