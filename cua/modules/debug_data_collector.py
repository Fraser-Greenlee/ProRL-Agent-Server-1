import asyncio
import copy
import json
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
from cua.modules.debug_uitars_actor import UITarsActor
from cua.modules.debug_planner import Planner
# from cua.modules.module_parser_controller import ParserController
from openhands.core.logger import openhands_logger
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime

from cua.modules.debug_env_controller import EnvController
from cua.modules.util import load_persona_dataset, load_osworld_setup_list, save_image, load_example_instructions

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
        self.planner = Planner(args)
        self.actor = UITarsActor(args)

        self.vm_image_path = args.vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in self.vm_image_path else 'windows'

        self.max_steps_per_trajectory = args.max_steps_per_trajectory
        self.max_steps_per_goal = args.max_steps_per_goal

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

        # load osworld setup list
        self.osworld_setup_list = load_osworld_setup_list(args.osworld_setup_path, logger)

        # load example instructions
        self.example_instructions = load_example_instructions(args.example_instructions_path, logger)

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

    @staticmethod
    def save_trajectory(trajectory: Dict, trajectory_save_dir: Path):
        """
        Save trajectory data to trajectory_save_dir as json format.
        """
        trajectory_to_save = copy.deepcopy(trajectory)
        for step in trajectory_to_save["steps"]:
            for action in step['actions']:
                action.pop("screenshot_base64")

        with open(trajectory_save_dir / "trajectory.json", "w") as f:
            json.dump(trajectory_to_save, f, indent=4)

        logger.debug(f"✓ [save_trajectory] Trajectory saved to {str(trajectory_save_dir / 'trajectory.json')}.")

    async def submit_trajectory_job(self, trajectory_idx: int):
        # Create unique IDs
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}--{trajectory_idx:03d}"
        trajectory_id = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}--{trajectory_idx:03d}"
        trajectory_save_dir = Path(__file__).parent.parent / f"trajectories/{trajectory_id}"
        os.makedirs(trajectory_save_dir)

        # Sample osworld setup for this trajectory
        # First entry is None so you can skip setup if needed
        osworld_setup_ready, osworld_setup = False, None
        while not osworld_setup_ready:
            osworld_setup = random.choice(self.osworld_setup_list)
            if any("VLC_VERBOSE=-1 vlc --no-audio --no-video-title-show" in config['parameters'].get("command", "")
                   for config in osworld_setup['config']):
                # sometimes
               continue
            else:
                osworld_setup_ready = True

        runtime = await EnvController.initialize_runtime(job_id, self.vm_image_path, self.os_type, osworld_setup)

        # set screen width and height here
        self.screen_width, self.screen_height = EnvController.get_screen_size(runtime)

        # initialize trajectory - originally created by SyntheticDataGenerator.collect_trajectory and returned by the function
        trajectory = {
            'trajectory_id': trajectory_id,
            'metadata': {
                'vm_image': self.vm_image_path,
                'screen_size': f"{self.screen_width}x{self.screen_height}",
            },
            'goal': None,
            'steps': [],
        }

        # -- "steps" will be a list of -- #
        # {
        #   "subgoal": str, "subgoal_intent": str,
        #   "actions": a list of
        #   {
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
        #       },
        #   },
        # }

        # get the initial screenshot and save it
        time.sleep(3.0)  # wait for the UI to update
        screenshot_bytes = EnvController.get_screenshot(runtime)
        image_filename = trajectory_save_dir / f"0-0.png"
        save_image(screenshot_bytes, image_filename, logger)

        # generate goal in a separate loop
        prev_requirements = []  # will be a list of tuple [("condition 1", "verdict 1"), ...]
        while len(prev_requirements) < 10:
            # sample example goals (instructions)
            example_goals = random.sample(self.example_instructions, 1)  # for now, we sample 1 example goal
            goal, requirements = self.planner.generate_goal_with_long_horizon(
                screenshot_bytes, osworld_setup["config"], example_goals, prev_requirements,
            )

            # todo implement the verification mechanism for goal achievability using requirements
            if goal:
                trajectory['goal'] = goal
                break

        # reverting back to original goal - action
        while sum(len(s['actions']) for s in trajectory['steps']) < self.max_steps_per_trajectory:
            subgoal_idx = len(trajectory['steps'])
            prev_subgoal_intents = [g['subgoal_intent'] for g in trajectory['steps']]
            prev_subgoals = [g['subgoal'] for g in trajectory['steps']]
            prev_actor_infos = [g["actions"][-1]["action_generation"]["thought"] for g in trajectory['steps']]

            subgoal_intent, subgoal = self.planner.generate_subgoal(
                screenshot_bytes, trajectory['goal'], prev_subgoal_intents, prev_subgoals, prev_actor_infos
            )

            if subgoal.lower().strip() == "done":
                # the high-level goal is achieved.
                break
            elif subgoal.lower().strip() == "impossible":
                # the high-level goal is impossible to achieve due to discrepancy with the environment
                # (e.g., asking for non-existing files)
                break

            # initialize steps_for_this_goal: a container to store trajectory for this goal
            step_for_this_subgoal = {
                "subgoal": subgoal,
                "subgoal_intent": subgoal_intent,
                "actions": []
            }
            while len(step_for_this_subgoal['actions']) < self.max_steps_per_goal:
                history_images = [s['screenshot_base64'] for s in step_for_this_subgoal['actions']]
                history_responses = [s['action_generation']['generation'] for s in step_for_this_subgoal['actions']]

                # generate action
                action_generation_result = self.actor.generate_action(
                    subgoal, screenshot_bytes, history_images, history_responses
                )

                if action_generation_result is None:
                    # UI-TARS action generation failed (failed to meet the requirement)
                    break
                else:
                    pyautogui_command = action_generation_result["pyautogui_command"]
                    action_generation = action_generation_result["action_generation"]


                    # execute action
                    EnvController.execute_pyautogui_command(runtime, pyautogui_command)

                    # wait for UI to update
                    time.sleep(3.0)

                    # update steps_for_this_goal with the current screenshot & action_dict_list
                    step_for_this_subgoal['actions'].append({
                        "screenshot": str(image_filename.absolute()),
                        "screenshot_base64": bytes_to_base64(screenshot_bytes),
                        "pyautogui_command": pyautogui_command,
                        "action_generation": action_generation,
                    })

                    # save new screenshot
                    screenshot_bytes = EnvController.get_screenshot(runtime)
                    image_filename = trajectory_save_dir / f"{subgoal_idx}-{len(step_for_this_subgoal['actions'])}.png"
                    save_image(screenshot_bytes, image_filename, logger)

                    # if the executed action involves "finished", break the action generation loop
                    # we still need to save a new screenshot since the pyautogui_command might involve actions other than
                    # "finished".
                    if any(action_dict["action_type"] == "finished" for action_dict in action_generation["parsed_actions"]):
                        break

            # save the completed steps
            trajectory['steps'].append(step_for_this_subgoal)

            # save the trajectory after each sub-goal (we re-write on the same file)
            self.save_trajectory(trajectory, trajectory_save_dir)

            # todo for debugging - remove below
            import copy
            steps = copy.deepcopy(trajectory['steps'])
            for step in steps:
                for action in step['actions']:
                    action.pop("screenshot_base64")
            ipdb.set_trace()
            pass

        # full trajectory generation done
        self.save_trajectory(trajectory, trajectory_save_dir)
        ipdb.set_trace()
        pass

