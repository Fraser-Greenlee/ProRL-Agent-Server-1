import logging
import os
import random
import threading
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import ipdb
from openhands.core.logger import openhands_logger
from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime

from cua.modules.debug_env_controller import EnvController
from cua.modules.util import load_persona_dataset, load_osworld_setup_list


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
        self.vm_image_path = args.vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in self.vm_image_path else 'windows'

        # will be later set in the first observation
        self.screen_width = None
        self.screen_height = None

        # Default screen dimensions based on OS type
        if self.os_type == 'windows':
            self.default_screen_width, self.default_screen_height = 1280, 800
        else:
            self.default_screen_width, self.default_screen_height = 1920, 1080

        # set model name
        self.explorer_model_name = ''  # todo
        self.parser_model_name = ''  # todo

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
                'agent_model': self.explorer_model_name,
                'parser_model': self.parser_model_name,
                'screen_size': f"{self.screen_width}x{self.screen_height}",
                'persona': persona
            },
            'steps': [],
        }

        initial_image_path = trajectory_save_dir / "0.png"
        screenshot = EnvController.get_and_save_screenshot(job_details.runtime, initial_image_path)

        # todo utilize screenshot to generate actions
        ipdb.set_trace()
        pass






