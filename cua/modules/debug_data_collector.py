import asyncio
import copy
import logging
import os
import random
import threading
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import ipdb

from cua.debug.util import bytes_to_base64
from cua.modules.module_uitars_controller import UITarsController
from cua.modules.prev_openai_controller import OpenAIController
# from cua.modules.module_parser_controller import ParserController
from openhands.core.logger import openhands_logger
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime

from cua.modules.debug_env_controller import EnvController
from cua.modules.util import load_persona_dataset, load_osworld_setup_list, save_image

# Create a child logger
logger = openhands_logger.getChild('data_controller')
logger.setLevel(logging.DEBUG)


@dataclass
class TrajectoryJobDetails:
    """Details for a single trajectory collection job."""
    job_id: str = ''
    trajectory_id: str = ''
    persona: Optional[Dict[str, Any]] = None
    runtime: Optional[OSWorldSingularityRuntime] = None
    osworld_setup: Optional[Dict[str, Any]] = None
    trajectory_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    event: Optional[threading.Event] = None
    completed: bool = False


class DataCollector:
    """
    Class managing actual workflow to collect trajectory data.
    Orchestrates usage of EnvController and OpenAIWrapper.
    Excluded asyncio / threading for debugging purposes.
    """
    def __init__(self, args: Namespace):
        self.openai_controller = OpenAIController(args)
        self.uitar_controller = UITarsController(args)

        self.vm_image_path = args.vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in self.vm_image_path else 'windows'

        self.max_steps_per_trajectory = args.max_steps_per_trajectory
        self.max_steps_per_goal = args.max_steps_per_goal

        # model name and nodes
        self.explorer_node = ''  # todo
        self.parser_node = ''  # todo
        self.explorer_model_name = ''  # todo

        # will be later set in the first observation
        self.screen_width = None
        self.screen_height = None

        # Default screen dimensions based on OS type
        if self.os_type == 'windows':
            self.default_screen_width, self.default_screen_height = 1280, 800
        else:
            self.default_screen_width, self.default_screen_height = 1920, 1080

        # load persona dataset if args.persona_dataset_path is set
        self.persona_dfs, self.persona_df_weights = load_persona_dataset(args.persona_dataset_path, logger)

        self.osworld_setup_list = load_osworld_setup_list(args.osworld_setup_path, logger)


    def sample_persona(self) -> Optional[Dict[str, Any]]:
        """
        Sample a random persona from the dataset.
        Uses weighted random selection across memory-mapped dataframes.

        Returns:
            Dictionary containing persona information, or None if dataset not loaded
        """
        assert len(self.persona_dfs) > 0 and len(self.persona_df_weights) > 0, "`persona_dfs` should not be empty."

        # Randomly select a dataframe (weighted by number of records)
        selected_df = random.choices(self.persona_dfs, weights=self.persona_df_weights, k=1)[0]
        age = 1

        persona_info = None

        while age < 18:
            # Sample random persona from the selected dataframe
            persona_record = selected_df.sample(n=1).iloc[0].to_dict()

            # Extract key fields for goal generation
            persona_info = {
                'professional': persona_record.get('professional_persona', ''),
                'hobbies': persona_record.get('hobbies_and_interests', ''),
                'occupation': persona_record.get('occupation', ''),
                'age': persona_record.get('age', ''),
                'education': persona_record.get('education_level', ''),
                'city': persona_record.get('city', ''),
                'state': persona_record.get('state', ''),
                'skills': persona_record.get('skills_and_expertise', ''),
                'interests_list': eval(persona_record.get('hobbies_and_interests_list', '')),  # should be a list
                'career_goals': persona_record.get('career_goals_and_ambitions', ''),
            }

            age = persona_record['age']

        return persona_info

    async def submit_trajectory_job(self, trajectory_idx: int):
        # Create unique IDs
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}--{trajectory_idx:03d}"
        trajectory_id = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}--{trajectory_idx:03d}"
        trajectory_save_dir = Path(__file__).parent.parent / f"trajectories/{trajectory_id}"
        os.makedirs(trajectory_save_dir)

        # Sample persona for this trajectory
        persona = self.sample_persona()

        # Sample osworld setup for this trajectory
        # First entry is None so you can skip setup if needed
        osworld_setup = random.choice(self.osworld_setup_list)

        # Create job details - we will use this as data interface for calling different functions here,
        # but later, the job_details will be stored into self.job_details
        job_details = TrajectoryJobDetails(
            job_id=job_id,
            trajectory_id=trajectory_id,
            persona=persona,
            osworld_setup=osworld_setup,
            event=threading.Event(),
        )

        job_details.runtime = await EnvController.initialize_runtime(job_id, self.vm_image_path, self.os_type, osworld_setup)

        # set screen width and height here
        self.screen_width, self.screen_height = EnvController.get_screen_size(job_details.runtime)

        # initialize trajectory - originally created by SyntheticDataGenerator.collect_trajectory and returned by the function
        trajectory = {
            'trajectory_id': trajectory_id,
            'metadata': {
                'vm_image': self.vm_image_path,
                'explorer_model': self.explorer_model_name,
                'screen_size': f"{self.screen_width}x{self.screen_height}",
                'persona': persona
            },
            'goals': [],
        }

        # -- "goals" will be a list of -- #
        # {
        # "goal": str, "goal_intent": str,
        # "actions": a list of
        # {
        #       "screenshot": string filename of the image, screenshot before the action
        #       "screenshot_base64": str,
        #       "pyautogui_command": a string command to send to EnvController
        #       "action_generation": {
        #           "generation": str (raw generation from UI-TARS),
        #           "reflection": str,
        #           "thought": str,
        #           "parsed_actions": list of {
        #               "action_type": str (e.g., "click"),
        #               "action_inputs": dict (e.g., {"start_box": "[0.089, 0.424]"}),
        #           },
        #        }
        # }

        # get the initial screenshot and save it
        time.sleep(3.0)  # wait for the UI to update
        screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
        image_filename = trajectory_save_dir / f"0-0.png"
        save_image(screenshot_bytes, image_filename, logger)

        while sum(len(g['actions']) for g in trajectory['goals']) < self.max_steps_per_trajectory:
            goal_idx = len(trajectory['goals'])
            prev_goal_intents = [g['goal_intent'] for g in trajectory['goals']]
            prev_goals = [g['goal'] for g in trajectory['goals']]

            # generate goal
            goal_intent, goal = self.openai_controller.generate_goal_with_persona(
                screenshot_bytes, persona, prev_goal_intents, prev_goals
            )

            # initialize steps_for_this_goal: a container to store trajectory for this goal
            steps_for_this_goal = {
                "goal": goal,
                "goal_intent": goal_intent,
                "actions": []
            }
            while len(steps_for_this_goal['actions']) < self.max_steps_per_goal:
                history_images = [s['screenshot_base64'] for s in steps_for_this_goal['actions']]
                history_responses = [s['action_generation']['generation'] for s in steps_for_this_goal['actions']]

                # generate action
                action_generation_result = self.uitar_controller.generate_action(
                    goal, screenshot_bytes, history_images, history_responses
                )
                pyautogui_command = action_generation_result["pyautogui_command"]
                action_generation = action_generation_result["action_generation"]

                # execute action
                EnvController.execute_pyautogui_command(job_details.runtime, pyautogui_command)

                # wait for UI to update
                time.sleep(4.5)

                # update steps_for_this_goal with the current screenshot & action_dict_list
                steps_for_this_goal['actions'].append({
                    "screenshot": str(image_filename.absolute()),
                    "screenshot_base64": bytes_to_base64(screenshot_bytes),
                    "pyautogui_command": pyautogui_command,
                    "action_generation": action_generation,
                })

                # save new screenshot
                screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
                image_filename = trajectory_save_dir / f"{goal_idx}-{len(steps_for_this_goal['actions'])}.png"
                save_image(screenshot_bytes, image_filename, logger)

                # if the executed action involves "finished", break the action generation loop
                # we still need to save new screenshot since the pyautogui_command might involve actions other than
                # "finished".
                if any(action_dict["action_type"] == "finished" for action_dict in action_generation["parsed_actions"]):
                    break

            # save the completed steps
            trajectory['goals'].append(steps_for_this_goal)

            # for debugging, remove screenshot_base64
            import copy
            previous_steps = copy.deepcopy(steps_for_this_goal)
            previous_steps["actions"] = [{"screenshot": a['screenshot'], "pyautogui_command": a['pyautogui_command'], "action_generation": a['action_generation']} for a in previous_steps["actions"]]
            ipdb.set_trace() # to print out results:
            pass



