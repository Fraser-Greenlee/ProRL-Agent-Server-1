import asyncio
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

# from cua.modules.module_openai_controller import OpenAIController
from cua.modules.prev_openai_controller import OpenAIController
from cua.modules.module_parser_controller import ParserController
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
        self.parser_controller = ParserController(args)

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
            'steps': [],
        }

        # -- new version - cursor-moving focused -- #
        # steps will be a list of
        # {
        # "goal": str, "goal_intent": str,
        # "actions": [
        #   "screenshot": string filename of the image, screenshot before the action
        #   "action_dict": Dict,
        #   "converted_action_dict": Dict,
        #   "action_thought": str,
        #   ],
        # }
        # time.sleep(3.0)
        # screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
        # image_filename = trajectory_save_dir / f"0-0.png"
        # save_image(screenshot_bytes, image_filename, logger)
        #
        # cursor_x, cursor_y = EnvController.get_cursor_position(job_details.runtime)
        #
        # while sum(len(g['actions']) for g in trajectory['steps']) < self.max_steps_per_trajectory:
        #     goal_idx = len(trajectory['steps'])
        #     prev_goal_intents = [g['goal_intent'] for g in trajectory['steps']]
        #     prev_goals = [g['goal'] for g in trajectory['steps']]
        #
        #     # generate goal
        #     goal_intent, goal = self.openai_controller.generate_goal_with_persona(
        #         screenshot_bytes, persona, prev_goal_intents, prev_goals
        #     )
        #
        #     steps_for_this_goal = {
        #         "goal_intent": goal_intent,
        #         "goal": goal,
        #         "actions": []
        #     }
        #     while len(steps_for_this_goal['actions']) < self.max_steps_per_goal:
        #         # todo first perform cursor actions
        #         num_move_generation = 0
        #         while num_move_generation < 10:
        #             num_move_generation += 1
        #             logger.debug(f"num_move_generation for goal {goal_idx}, action {len(steps_for_this_goal['actions'])}: {num_move_generation}")
        #
        #             # generate and perform cursor-moving actions here
        #             # draw cursor bbox on the screenshot
        #             cursor_bbox = [cursor_x - 5, cursor_y - 5, cursor_x + 5, cursor_y + 5]
        #             screenshot_with_cursor = ParserController.prepare_image_with_som(
        #                 screenshot_bytes, [cursor_bbox], False
        #             )
        #             # todo remove this - for debugging, save screenshot
        #             image_filename = trajectory_save_dir / f"{goal_idx}-{len(steps_for_this_goal['actions'])}-{num_move_generation}-cursor.png"
        #             save_image(screenshot_with_cursor, image_filename, logger)
        #
        #             move_x, move_y = self.openai_controller.generate_cursor_moving_action(
        #                 screenshot_with_cursor, goal, cursor_x, cursor_y, self.screen_width, self.screen_height,
        #             )
        #
        #             # todo if move_x == 0 and move_y == 0, break without updating anything
        #             if move_x == 0 and move_y == 0:
        #                 break
        #             else:
        #                 # update cursor position
        #                 job_details.runtime.execute_vm_action({
        #                     "action_type": "MOVE_TO",
        #                     "parameters": {"x": cursor_x + move_x, "y": cursor_y + move_y}
        #                 })
        #                 time.sleep(0.5)
        #                 cursor_x, cursor_y = EnvController.get_cursor_position(job_details.runtime)
        #
        #                 # update screenshot
        #                 screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
        #
        #             ipdb.set_trace()
        #             pass
        #
        #         print(f"Movement done!")
        #         ipdb.set_trace()
        #         pass






        # -- previous version -- #
        # steps will be a list of
        # {
        # "goal": str, "goal_intent": str,
        # "actions": [
        #   "screenshot": string filename of the image, screenshot before the action
        #   "action_dict": Dict,
        #   "converted_action_dict": Dict,
        #   "action_thought": str,
        #   ],
        # }

        # get the initial screenshot and save it
        time.sleep(3.0)  # wait for the UI to update
        screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
        parsed_screenshot_bytes, bbox_list = self.parser_controller.parse_screenshot(screenshot_bytes)
        # todo we will have to save the unparsed image when not debugging
        # image_filename = trajectory_save_dir / f"{goal_idx}-0.png"
        # save_image(screenshot_bytes, image_filename, logger)
        image_filename = trajectory_save_dir / f"0-0.png"
        save_image(parsed_screenshot_bytes, image_filename, logger)

        # # get the initial ast - todo move this to EnvController if working
        action = OSWorldInteractiveAction(
            method='get_accessibility_tree',
            params={},
            thought='Getting UI accessibility tree'
        )
        ast_obs = job_details.runtime.run_action(action)
        simplified_ast = ast_obs.content[0]  # todo see if simplified_ast can work with X icon
        ipdb.set_trace()
        pass

        while sum(len(g['actions']) for g in trajectory['steps']) < self.max_steps_per_trajectory:
            goal_idx = len(trajectory['steps'])
            prev_goal_intents = [g['goal_intent'] for g in trajectory['steps']]
            prev_goals = [g['goal'] for g in trajectory['steps']]

            # generate goal
            goal_intent, goal = self.openai_controller.generate_goal_with_persona(
                parsed_screenshot_bytes, persona, prev_goal_intents, prev_goals
            )

            steps_for_this_goal = {
                "goal_intent": goal_intent,
                "goal": goal,
                "actions": []
            }
            while len(steps_for_this_goal['actions']) < self.max_steps_per_goal:
                prev_thoughts = [s['action_thought'] for s in steps_for_this_goal['actions']]
                prev_actions = [s['action_dict'] for s in steps_for_this_goal['actions']]

                # generate action
                action_thought, action_dict = self.openai_controller.generate_action(
                    parsed_screenshot_bytes, goal, prev_thoughts, prev_actions
                )

                try:
                    # convert action to runtime-executable format
                    converted_action_dict = EnvController.convert_action_dict(action_dict, bbox_list)

                    # execute action
                    if converted_action_dict is not None:  # converted_action_dict is None for when action is "done"
                        action_result = EnvController.execute_action(converted_action_dict, job_details.runtime)

                        # wait for the UI to update
                        time.sleep(4.5)  # todo check if this is asyncio-safe, or should we use await asyncio.sleep?

                    # save the current step
                    step_dict = {
                        "screenshot": str(image_filename.absolute()),
                        "action_thought": action_thought,
                        "action_dict": action_dict,
                        "converted_action_dict": converted_action_dict,
                    }
                    steps_for_this_goal['actions'].append(step_dict)

                    # update screenshot
                    screenshot_bytes = EnvController.get_screenshot(job_details.runtime)
                    # todo we will have to save this image when not debugging
                    # image_filename = trajectory_save_dir / f"{goal_idx}-{len(steps_for_this_goal['actions'])}.png"
                    # save_image(screenshot_bytes, image_filename, logger)

                    # parse the new screenshot
                    parsed_screenshot_bytes, bbox_list = self.parser_controller.parse_screenshot(screenshot_bytes)
                    # todo remove save_image below when not debugging
                    image_filename = trajectory_save_dir / f"{goal_idx}-{len(steps_for_this_goal['actions'])}.png"
                    save_image(parsed_screenshot_bytes, image_filename, logger)

                    # if the last action was "done", break the loop for the current goal
                    if converted_action_dict is None:
                        break

                    ipdb.set_trace()
                    pass

                except Exception as e:
                    logger.debug(f"Error in generating or executing action: {e}")

            # save the completed steps
            trajectory['steps'].append(steps_for_this_goal)




