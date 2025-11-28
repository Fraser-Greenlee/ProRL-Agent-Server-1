#!/usr/bin/env python3
"""
Synthetic Data Generator for Computer Use Trajectories

This script collects trajectories of an AI agent interacting with OSWorld:
1. Gets current screen state and accessibility tree
2. LLM imagines a local goal based on screen state
3. LLM selects and executes an appropriate action
4. Saves trajectory data (image, AST, simplified AST, goal, action)
5. Repeats for multiple steps

Author: OpenHands Team
Date: 2025
"""

import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

import pandas as pd
from openai import OpenAI
from PIL import Image
from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.events.observation import ErrorObservation
from openhands.runtime.impl.singularity.osworld_singularity_runtime import (
    OSWorldSingularityRuntime,
)
from openhands.storage import get_file_store
from openhands.utils.ast_process import simplify_accessibility_tree, simplify_json_accessibility_tree, simplify_json_to_xml
from openhands.utils.ast_process_win import simplify_windows_accessibility_tree
from openhands.agenthub.gui_agent.tools import OSWORLD_TOOLS
import time
from openhands.nvidia.os_world.controllers.setup import SetupController


TOOL_NAME_TO_ACTION_TYPE = {
    'click': 'CLICK',
    'rightClick': 'RIGHT_CLICK',
    'middleClick': 'MIDDLE_CLICK',
    'doubleClick': 'DOUBLE_CLICK',
    'tripleClick': 'TRIPLE_CLICK',
    'moveTo': 'MOVE_TO',
    'dragTo': 'DRAG_TO',
    'scroll': 'SCROLL',
    'hscroll': 'SCROLL',
    'write': 'TYPING',
    'press': 'PRESS',
    'hotkey': 'HOTKEY',
}

def fix_tool_schema(tool: dict) -> dict:
    # Create new clean tool
    new_tool = {
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"].get("description", ""),
        "parameters": tool["function"].get("parameters", {}),
    }

    return new_tool

class TrajectoryJobDetails:
    """Details for a single trajectory collection job."""
    def __init__(self):
        self.job_id: str = ''
        self.trajectory_id: str = ''
        self.persona: Optional[Dict[str, Any]] = None
        self.runtime: Optional[OSWorldSingularityRuntime] = None
        self.trajectory_data: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.event: Optional[threading.Event] = None
        self.completed: bool = False


class SyntheticDataGenerator:
    """Generates synthetic computer use trajectories using OSWorld and LLM."""
    
    def __init__(
        self,
        vm_image_path: str,
        output_dir: str = './trajectories',
        llm_base_url: str = 'http://localhost:8000/v1',
        llm_model: str = 'openai/gpt-oss-120b',
        max_steps_per_trajectory: int = 20,
        max_trajectories: int = 10,
        persona_dataset_path: Optional[str] = '/work/Projects/data/nemotron_data/data',
        osworld_setup_dataset_path: Optional[str] = '/root/OSWorld/osworld_test_nogdrive.json',
        max_parallel: int = 3,
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            vm_image_path: Path to the VM image file
            output_dir: Directory to save trajectories
            llm_base_url: Base URL for LLM API
            llm_model: Model name to use
            max_steps_per_trajectory: Maximum steps per trajectory
            max_trajectories: Maximum number of trajectories to collect
            persona_dataset_path: Path to nemotron persona dataset (parquet files)
            max_parallel: Maximum number of parallel trajectory collectors (default: 3)
        """
        self.vm_image_path = vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in vm_image_path else 'windows'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.max_trajectories = max_trajectories
        self.max_parallel = max_parallel
        

        self.llm_client = OpenAI(
            base_url=llm_base_url,
            api_key='EMPTY',  # vLLM ignores this by default
        )
        
        # Get OSWorld tool definition for vLLM
        # Convert to unified new schema
        self.osworld_tool = []
        # remove finish, wait, and fail tools
        for tool in OSWORLD_TOOLS[:-3]:
            self.osworld_tool.append(fix_tool_schema(tool))
        
        # Queue-based architecture (following async_server.py pattern)
        self.init_queue: queue.Queue[str] = queue.Queue()
        self.collect_queue: queue.Queue[str] = queue.Queue()
        self._init_workers: List[Optional[asyncio.AbstractEventLoop]] = []
        self._collect_workers: List[Optional[asyncio.AbstractEventLoop]] = []
        self._job_details: Dict[str, TrajectoryJobDetails] = {}
        self._job_details_lock = threading.RLock()
        self._server_running: bool = False
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Concurrency control: limit total concurrent runtimes
        self._runtime_semaphore = threading.Semaphore(max_parallel)
        self._active_runtime_count = 0
        self._active_runtime_lock = threading.Lock()
        
        self.screen_width = None
        self.screen_height = None
        
        # Default screen dimensions based on OS type
        if self.os_type == 'windows':
            self.default_screen_width, self.default_screen_height = 1280, 800
        else:
            self.default_screen_width, self.default_screen_height = 1920, 1080
        
        # Load persona dataset
        self.persona_dfs = []  # List of memory-mapped dataframes
        self.persona_df_weights = []  # Weights for sampling
        self.persona_dataset_path = persona_dataset_path
        if persona_dataset_path and os.path.exists(persona_dataset_path):
            self._load_persona_dataset()
        else:
            logger.warning(f"Persona dataset path not found: {persona_dataset_path}")
            logger.warning("  Continuing without persona-based goal generation")

        self.osworld_setup_dataset_path = osworld_setup_dataset_path
        if osworld_setup_dataset_path and os.path.exists(osworld_setup_dataset_path):
            self._load_osworld_setup_dataset()
        else:
            logger.warning(f"OSWorld setup dataset path not found: {osworld_setup_dataset_path}")
            logger.warning("  Continuing without OSWorld setup")
        
        logger.info(f"Initialized SyntheticDataGenerator")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  LLM: {llm_model} @ {llm_base_url}")
        logger.info(f"  Max steps per trajectory: {max_steps_per_trajectory}")
        logger.info(f"  Max trajectories: {max_trajectories}")
        logger.info(f"  Max parallel workers: {max_parallel}")
        if self.persona_dfs:
            logger.info(f"  Personas loaded: {sum(self.persona_df_weights):,} records")

        os.makedirs('/tmp/osworld_example', exist_ok=True)
    
    def _load_persona_dataset(self):
        """Load the nemotron persona dataset from parquet files."""
        logger.info(f"Loading persona dataset from {self.persona_dataset_path}...")
        
        try:
            # Load all parquet files
            parquet_files = list(Path(self.persona_dataset_path).glob('train-*.parquet'))
            
            if not parquet_files:
                logger.warning(f"No parquet files found in {self.persona_dataset_path}")
                return
            
            logger.info(f"  Found {len(parquet_files)} parquet files")
            
            # Load all parquet files using memory mapping for efficiency
            # Keep them as separate dataframes to maintain memory mapping benefits
            logger.info("  Loading files with memory mapping...")
            self.persona_dfs = []  # List of memory-mapped dataframes
            self.persona_df_weights = []  # Weights for random sampling
            total_records = 0
            
            for pf in parquet_files:
                df = pd.read_parquet(pf, memory_map=True)
                self.persona_dfs.append(df)
                self.persona_df_weights.append(len(df))
                total_records += len(df)
                logger.info(f"    Loaded {pf.name}: {len(df):,} records")
            
            logger.info(f"  ✓ Loaded {total_records:,} total persona records from {len(parquet_files)} files (memory-mapped)")
            logger.info(f"  Fields: {', '.join(self.persona_dfs[0].columns[:8])}...")
            
        except Exception as e:
            logger.error(f"Error loading persona dataset: {e}")
            self.persona_dfs = []
            self.persona_df_weights = []

    def _load_osworld_setup_dataset(self):
        """Load the osworld setup dataset from json files. Always have None entry indicating no setup."""
        logger.info(f"Loading osworld setup dataset from {self.osworld_setup_dataset_path}...")
        
        try:
            import json
            self.osworld_setup_dataset = [None]
            with open(self.osworld_setup_dataset_path, 'r') as f:
                for line in f.readlines():
                    self.osworld_setup_dataset.append(json.loads(line))
            logger.info(f"  ✓ Loaded {len(self.osworld_setup_dataset)} total osworld setup records")
            
        except Exception as e:
            logger.error(f"Error loading osworld setup dataset: {e}")
            self.osworld_setup_dataset = [None]
    
    def sample_persona(self) -> Optional[Dict[str, Any]]:
        """
        Sample a random persona from the dataset.
        Uses weighted random selection across memory-mapped dataframes.
        
        Returns:
            Dictionary containing persona information, or None if dataset not loaded
        """
        if not self.persona_dfs or sum(self.persona_df_weights) == 0:
            return None
        
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
            'interests_list': persona_record.get('hobbies_and_interests_list', ''),
            'career_goals': persona_record.get('career_goals_and_ambitions', ''),
        }
        
        return persona_info
    
    def start_workers(self):
        """Start init and collect worker threads."""
        if self._server_running:
            raise RuntimeError('Workers are already running')
        self._server_running = True
        
        # Create thread pool executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_parallel * 2  # init + collect workers
        )
        
        # Initialize worker lists
        self._init_workers = [None] * self.max_parallel
        self._collect_workers = [None] * self.max_parallel
        
        logger.info(f"Starting {self.max_parallel} init workers...")
        for i in range(self.max_parallel):
            self._executor.submit(self._run_init_worker_in_thread, i)
        
        logger.info(f"Starting {self.max_parallel} collect workers...")
        for i in range(self.max_parallel):
            self._executor.submit(self._run_collect_worker_in_thread, i)
        
        logger.info(f"✓ Workers started: {self.max_parallel} init + {self.max_parallel} collect")
    
    def stop_workers(self):
        """Stop all workers and cleanup."""
        if not self._server_running:
            return
        
        logger.info("Stopping workers...")
        self._server_running = False
        
        # Send stop signals to all queues
        for _ in range(self.max_parallel):
            try:
                self.init_queue.put_nowait('__STOP__')
                self.collect_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f"Error sending stop signal: {e}")
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logger.info("✓ Workers stopped")
    
    def _cleanup_job_runtime(self, runtime: OSWorldSingularityRuntime, job_id: str):
        """Cleanup runtime in background thread (following async_server.py pattern)."""
        def close():
            try:
                runtime.close()
                if hasattr(runtime, 'event_stream') and runtime.event_stream:
                    try:
                        runtime.event_stream.close()
                        logger.debug(f'Event stream closed for job {job_id}')
                    except Exception as e:
                        logger.warning(f'Error closing event stream for job {job_id}: {e}')
                time.sleep(0.1)  # Brief pause for cleanup
            except Exception as e:
                logger.error(f'Error cleaning up runtime for job {job_id}: {e}')
        
        # Run cleanup in background thread
        t = threading.Thread(target=close, daemon=True)
        t.start()
    
    async def _init_worker(self, worker_id: int):
        """Init worker: initializes runtimes for trajectory jobs."""
        logger.info(f"[init-worker-{worker_id}] Started")
        
        while True:
            # Get job from queue
            job_id = await asyncio.to_thread(self.init_queue.get)
            
            # Check for stop signal
            if job_id == '__STOP__':
                logger.info(f"[init-worker-{worker_id}] Received stop signal, exiting")
                self.init_queue.task_done()
                break
            
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(f"[init-worker-{worker_id}] Job {job_id} not found")
                    self.init_queue.task_done()
                    continue
            
            # WAIT for available slot (blocks if max_parallel runtimes already active)
            logger.info(f"[init-worker-{worker_id}] Waiting for runtime slot...")
            await asyncio.to_thread(self._runtime_semaphore.acquire)
            
            with self._active_runtime_lock:
                self._active_runtime_count += 1
            
            logger.info(f"[init-worker-{worker_id}] Runtime slot acquired ({self._active_runtime_count}/{self.max_parallel}), initializing {job_id}")
            
            try:
                # Create runtime configuration
                config = OpenHandsConfig()
                config.runtime = 'osworld'
                config.sandbox.base_container_image = 'ubuntu:24.04'
                config.sandbox.run_as_fakeroot = True
                # runtime_container_image will be built automatically if None
                config.sandbox.runtime_container_image = None  # Trigger auto-build
                
                # Unique event stream per trajectory
                file_store = get_file_store('local', f'/tmp/synthetic_data_gen_{job_id}')
                event_stream = EventStream(sid=job_id, file_store=file_store)
                
                # Validate VM image path exists
                if not os.path.exists(self.vm_image_path):
                    raise RuntimeError(f"VM image not found: {self.vm_image_path}")
                
                logger.info(f"[init-worker-{worker_id}] Creating runtime for {job_id}")
                logger.info(f"[init-worker-{worker_id}]   VM image: {self.vm_image_path}")
                logger.info(f"[init-worker-{worker_id}]   Base image: {config.sandbox.base_container_image}")
                
                # Create runtime
                runtime = OSWorldSingularityRuntime(
                    config=config,
                    event_stream=event_stream,
                    sid=job_id,
                    os_type=self.os_type,
                    vm_image_path=self.vm_image_path,
                    attach_to_existing=False,
                )
                
                logger.info(f"[init-worker-{worker_id}] Runtime object created, connecting to VM...")
                
                # Connect to VM
                await runtime.connect()
                logger.info(f"[init-worker-{worker_id}] ✓ Runtime initialized and connected for {job_id}")
                logger.info(f"[init-worker-{worker_id}]   VM URL: {runtime.osworld_vm_url if hasattr(runtime, 'osworld_vm_url') else 'N/A'}")

                if job_details.osworld_setup and self.os_type == 'linux':
                    logger.info(f"[init-worker-{worker_id}] Setting up OSWorld...")
                    setup_controller = SetupController(
                        vm_ip="127.0.0.1",
                        server_port=runtime._vm_server_port,
                        chromium_port=runtime._chromium_port,
                        cache_dir="/tmp/osworld_example", # might need to be changed to a unique directory for each job
                        client_password="password",
                        runtime=runtime  
                    )
                    await setup_controller.setup(job_details.osworld_setup['config'])
                    logger.info(f"[init-worker-{worker_id}] ✓ OSWorld setup completed")
                else:
                    logger.info(f"[init-worker-{worker_id}] No OSWorld setup provided")
                
                # Store runtime in job details
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].runtime = runtime
                        # Move to collect queue
                        self.collect_queue.put(job_id)
                
            except Exception as e:
                logger.error(f"[init-worker-{worker_id}] Failed to init runtime for {job_id}: {e}")
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].error = str(e)
                        self._job_details[job_id].event.set()
                
                # Release semaphore on failure (no runtime to cleanup later)
                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1
                logger.info(f"[init-worker-{worker_id}] Runtime slot released due to error ({self._active_runtime_count}/{self.max_parallel})")
            
            finally:
                self.init_queue.task_done()
    
    async def _collect_worker(self, worker_id: int):
        """Collect worker: collects trajectories using initialized runtimes."""
        logger.info(f"[collect-worker-{worker_id}] Started")
        
        while True:
            # Get job from queue
            job_id = await asyncio.to_thread(self.collect_queue.get)
            
            # Check for stop signal
            if job_id == '__STOP__':
                logger.info(f"[collect-worker-{worker_id}] Received stop signal, exiting")
                self.collect_queue.task_done()
                break
            
            with self._job_details_lock:
                job_details = self._job_details.get(job_id)
                if job_details is None:
                    logger.warning(f"[collect-worker-{worker_id}] Job {job_id} not found")
                    self.collect_queue.task_done()
                    continue
            
            logger.info(f"[collect-worker-{worker_id}] Collecting trajectory {job_details.trajectory_id}")
            
            try:
                # Collect trajectory (saves incrementally after each step)
                trajectory_data = await self.collect_trajectory(
                    job_details.trajectory_id,
                    job_details.runtime,
                    job_details.persona
                )
                
                # Final save with verbose logging
                self.save_trajectory(trajectory_data, verbose=True)
                
                # Store results
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].trajectory_data = trajectory_data
                        self._job_details[job_id].completed = True
                
                logger.info(f"[collect-worker-{worker_id}] ✓ Trajectory {job_details.trajectory_id} completed")
                
            except Exception as e:
                logger.error(f"[collect-worker-{worker_id}] Error collecting {job_details.trajectory_id}: {e}")
                import traceback
                traceback.print_exc()
                
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].error = str(e)
            
            finally:
                # Cleanup runtime after trajectory completes
                if job_details.runtime:
                    logger.info(f"[collect-worker-{worker_id}] Cleaning up runtime for {job_id}")
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    with self._job_details_lock:
                        if job_id in self._job_details:
                            self._job_details[job_id].runtime = None
                
                # Release semaphore slot (allow next runtime to be created)
                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1
                logger.info(f"[collect-worker-{worker_id}] Runtime slot released ({self._active_runtime_count}/{self.max_parallel})")
                
                # Signal completion
                if job_details.event:
                    job_details.event.set()
                
                self.collect_queue.task_done()
    
    def _run_init_worker_in_thread(self, worker_id: int):
        """Run init worker in its own thread with event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._init_workers[worker_id] = loop
        
        try:
            loop.run_until_complete(self._init_worker(worker_id))
        finally:
            try:
                loop.close()
                logger.debug(f'Event loop closed for init worker {worker_id}')
            except Exception as e:
                logger.warning(f'Error closing event loop for init worker {worker_id}: {e}')
    
    def _run_collect_worker_in_thread(self, worker_id: int):
        """Run collect worker in its own thread with event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._collect_workers[worker_id] = loop
        
        try:
            loop.run_until_complete(self._collect_worker(worker_id))
        finally:
            try:
                loop.close()
                logger.debug(f'Event loop closed for collect worker {worker_id}')
            except Exception as e:
                logger.warning(f'Error closing event loop for collect worker {worker_id}: {e}')
    
    def get_current_state(self, runtime: OSWorldSingularityRuntime) -> Dict[str, Any]:
        """
        Get the current screen state including screenshot, accessibility tree, and cursor position.
        
        Args:
            runtime: Runtime instance to use for state capture
            
        Returns:
            Dictionary with screenshot, AST, simplified AST, and cursor position
        """
        logger.info("Getting current screen state...")
        
        # Get screenshot directly (not as action)
        screenshot_bytes = runtime.get_vm_screenshot()
        if screenshot_bytes:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        else:
            screenshot_b64 = ''
        
        # Get cursor position
        cursor_x, cursor_y = 0, 0
        try:
            # Get cursor position via pyautogui in the VM
            action = OSWorldInteractiveAction(
                method='execute_python_command',
                params={'command': 'import pyautogui; pos = pyautogui.position(); print(f"{pos.x},{pos.y}")'},
                thought='Getting cursor position'
            )
            cursor_obs = runtime.run_action(action)
            if cursor_obs and cursor_obs.content:
                coords = cursor_obs.content.strip().split(',')
                if len(coords) == 2:
                    cursor_x = int(coords[0])
                    cursor_y = int(coords[1])
                    logger.info(f"Cursor position: ({cursor_x}, {cursor_y})")
        except Exception as e:
            logger.warning(f"Could not get cursor position: {e}")
        
        # Get accessibility tree
        action = OSWorldInteractiveAction(
            method='get_accessibility_tree',
            params={},
            thought='Getting UI accessibility tree'
        )
        ast_obs = runtime.run_action(action)
        
        # Parse AST (it's returned as JSON with 'AT' key containing XML)
        try:
            ast_data = json.loads(ast_obs.content)
            ast_xml = ast_data.get('AT', '')
        except:
            ast_xml = ast_obs.content
        
        # Simplify AST for LLM using ast_process simplifier
        # This returns clean representation with center coordinates and bounding boxes
        raw_ast = None  # Store the raw AST for debugging/logging
        if self.os_type == 'windows':
            simplified_ast, (self.screen_width, self.screen_height) = simplify_windows_accessibility_tree(ast_xml)
            raw_ast = ast_xml
        else:
            # Use JSON simplifier for Linux (new format)
            simplified_ast_dict, (self.screen_width, self.screen_height) = simplify_json_accessibility_tree(ast_xml)
            # Convert to XML string for consistency with existing code
            simplified_ast, _ = simplify_json_to_xml(simplified_ast_dict)
            raw_ast = json.dumps(ast_xml)  # Store JSON as string for raw_ast
        
        # Normalize cursor position to [0, 1] range like other coordinates
        width = self.screen_width if self.screen_width is not None else self.default_screen_width
        height = self.screen_height if self.screen_height is not None else self.default_screen_height
        
        normalized_cursor_x = cursor_x / width if width > 0 else 0
        normalized_cursor_y = cursor_y / height if height > 0 else 0
        
        return {
            'screenshot': screenshot_b64,
            'ast_xml': raw_ast,  # Raw AST for debugging (XML string or JSON string)
            'simplified_ast': simplified_ast,
            'cursor_position': {'x': round(normalized_cursor_x, 3), 'y': round(normalized_cursor_y, 3)},
            'timestamp': time.time()
        }
    
    def _truncate_observation(self, observation: Dict[str, Any], max_chars: int = 20000) -> Dict[str, Any]:
        """
        Truncate observation data if it's too large to prevent context overflow.
        
        Args:
            observation: Observation dictionary to potentially truncate
            max_chars: Maximum characters allowed in JSON representation
            
        Returns:
            Truncated observation dictionary
        """
        obs_json = json.dumps(observation)
        
        if len(obs_json) <= max_chars:
            return observation
        
        logger.warning(f"Observation too large ({len(obs_json):,} chars), truncating to {max_chars:,} chars")
        
        # Create truncated copy
        truncated_obs = observation.copy()
        
        # Truncate the most verbose fields
        if 'Current Screen State' in truncated_obs:
            state_str = str(truncated_obs['Current Screen State'])
            if len(state_str) > max_chars // 2:
                truncated_obs['Current Screen State'] = state_str[:max_chars // 2] + "\n... [TRUNCATED]"
        
        return truncated_obs
    
    def save_screenshot(self, screenshot_content: str, trajectory_id: str, step: int) -> str:
        """
        Save screenshot to file.
        
        Args:
            screenshot_content: Screenshot content (may be base64 or file path)
            trajectory_id: Trajectory identifier
            step: Step number
            
        Returns:
            Path to saved screenshot
        """
        # Create trajectory directory
        traj_dir = self.output_dir / trajectory_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        # Save screenshot
        screenshot_path = traj_dir / f"screenshot_step_{step:03d}.png"
        
        # Handle different content formats
        try:
            # Try base64 decode
            screenshot_b64 = screenshot_content
            if screenshot_b64.startswith('base64:'):
                screenshot_b64 = screenshot_b64[7:]
            
            screenshot_data = base64.b64decode(screenshot_b64)
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_data)
        except Exception as e:
            # If base64 decode fails, check if it's a file path
            if os.path.exists(screenshot_content):
                # Copy from existing file
                import shutil
                shutil.copy(screenshot_content, screenshot_path)
            else:
                logger.warning(f"Could not decode screenshot, saving content as text for debugging")
                # Save raw content for debugging
                with open(screenshot_path.with_suffix('.txt'), 'w') as f:
                    f.write(f"Error: {e}\n")
                    f.write(f"Content preview: {screenshot_content[:500]}\n")
                # Create empty image
                from PIL import Image
                img = Image.new('RGB', (800, 600), color='gray')
                img.save(screenshot_path)
        
        return str(screenshot_path.relative_to(self.output_dir))
    
    def generate_goal(
        self,
        state: Dict[str, Any],
        historical_goals: List[str],
        persona: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Use LLM to generate a sub-goal based on current state and persona.
        
        Args:
            state: Current screen state with simplified AST
            historical_goals: List of previously generated goals
            persona: Optional persona information to guide goal generation
            
        Returns:
            Goal string, or None if LLM decides to stop
        """
        logger.info("Generating sub-goal with LLM...")
        
        # Format historical goals
        goals_history = ""
        if historical_goals:
            goals_history = "\nPrevious goals you've pursued:\n"
            for i, goal in enumerate(historical_goals): 
                goals_history += f"{i+1}. {goal}\n"
            goals_history += "\nTry to finish the previous goal. e.g. if you clicked on URL field, your next goal can be typing the URL\n"
        
        # Prepare persona context
        persona_context = ""
        if persona:
            persona_context = f"\n**Your Persona Context:**\n"
            
            # Add occupation and demographics
            if persona.get('occupation'):
                occupation = persona['occupation'].replace('_', ' ').title()
                persona_context += f"- Occupation: {occupation}"
                if persona.get('age'):
                    persona_context += f" (age {persona['age']})"
                if persona.get('city') and persona.get('state'):
                    persona_context += f" from {persona['city']}, {persona['state']}"
                persona_context += "\n"
            
            # Add interests/hobbies (parse from list string)
            if persona.get('interests_list'):
                try:
                    import ast
                    interests = ast.literal_eval(persona['interests_list'])
                    if interests and len(interests) > 0:
                        # Sample 3-5 interests for variety
                        sample_interests = random.sample(interests, min(5, len(interests)))
                        persona_context += f"- Interests: {', '.join(sample_interests)}\n"
                except:
                    pass
            
            # Add brief professional/hobby description (truncated)
            if persona.get('professional'):
                prof_text = persona['professional'][:200].strip()
                if len(persona['professional']) > 200:
                    # Find last complete sentence
                    last_period = prof_text.rfind('.')
                    if last_period > 100:
                        prof_text = prof_text[:last_period+1]
                persona_context += f"- Work style: {prof_text}\n"
            
            if persona.get('hobbies'):
                hobby_text = persona['hobbies'][:200].strip()
                if len(persona['hobbies']) > 200:
                    last_period = hobby_text.rfind('.')
                    if last_period > 100:
                        hobby_text = hobby_text[:last_period+1]
                persona_context += f"- Personal interests: {hobby_text}\n"
            
            persona_context += "\nGenerate goals that align with this persona's background, interests, and typical activities.\n"
        
        # Prepare instructions (general guidelines)
        instructions = """You are an AI agent exploring a Ubuntu desktop environment.

Your task is to imagine ONE reasonable sub-goal you could achieve based on the current screen state.

Respond with a single, specific goal."""
        
        # Prepare user message with actual state data
        cursor_info = state.get('cursor_position', {})
        cursor_text = f"**Current Cursor Position:** ({cursor_info.get('x', 'unknown')}, {cursor_info.get('y', 'unknown')})\n\n" if cursor_info else ""
        
        user_message = f"""{cursor_text}{persona_context}Current Screen State:
{state['simplified_ast']}
{goals_history}

Guidelines:
- Don't ask clarification questions - just generate a simple goal
- A larger objective is irrelevant. We just need a local sub-goal to navigate the system. Don't ask for it!
- Choose realistic, achievable goals from visible UI elements
- Goals should be specific and actionable (e.g., "Type 'news' in search box", "Open Google Chrome") 
- Goals must be atomic - ONE action at a time. Click is one goal, type is another goal.
- Be curious and explore different parts of the system
- If the same goal is generated multiple times, it means you have been stuck in a loop. Try to backtrack and generate a different goal.
- Generate coherent goal sequences (e.g., if browser is open, search for something)
- Don't do random app switches in your goals
- If you cannot find some file in your previous goals, most likely it doesn't exist and hallucinated. Don't try to find it again and generate a different goal.
- Goals must be achievable with current screen state
{('- If persona context is provided, generate goals aligned with their interests, occupation, and typical activities' if persona else '')}

What specific sub-goal would you like to achieve next based on the visible elements{' and your persona' if persona else ''}?"""
        
        try:
            # Use responses.create API (vLLM)
            response = self.llm_client.responses.create(
                model=self.llm_model,
                instructions=instructions,
                input=[{"role": "user", "content": user_message}],
                reasoning={"effort": "high"},
            )
            
            goal = response.output_text.strip() if hasattr(response, 'output_text') else ""
            
            if not goal:
                logger.warning("LLM returned empty goal")
                return None
            
            # Check if agent wants to stop
            if any(word in goal.lower() for word in []):
                logger.info(f"Agent decided to stop: {goal}")
                return None
            
            logger.info(f"Generated goal: {goal}")
            return goal.strip()
        
        except Exception as e:
            logger.error(f"Error generating goal: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_action(
        self,
        state: Dict[str, Any],
        goal: str,
        steps: List[Dict[str, Any]],
        trajectory_id: str,
        runtime: OSWorldSingularityRuntime,
        max_tool_loops: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Use LLM to select an action based on the goal and current state.
        Uses tool calling loop until final answer (following client_test.py pattern).
        
        Args:
            state: Current screen state with simplified AST
            goal: The sub-goal to achieve
            steps: List of step data
            trajectory_id: Trajectory identifier
            runtime: Runtime instance to execute actions on
            max_tool_loops: Maximum tool call iterations
            
        Returns:
            List of action dictionaries, or None if failed
        """
        logger.info(f"Generating action for goal: {goal}")
        
        # Add cursor position info
        cursor_info = state.get('cursor_position', {})
        cursor_text = f"**Current Cursor Position:** ({cursor_info.get('x', 'unknown')}, {cursor_info.get('y', 'unknown')})\n\n" if cursor_info else ""
        
        # Prepare instructions for vLLM responses API
        instructions = f"""You are an AI agent controlling a Ubuntu desktop environment.

Available actions:
1. Type text: execute_action with ```write``` action_type
2. Click elements: execute_action with ```click``` action_type  
3. Hotkey keys: execute_action with ```hotkey``` action_type
4. You also have access to other GUI tools.

Use the osworld tool to interact. Look at the screen state and cursor position, select the appropriate action.
Use exact coordinates from the simplified AST."""

        user_content = f"""{cursor_text}Current Screen State:
{state['simplified_ast']}

Goal: {goal}

Select the appropriate action using the osworld tool."""
        
        try:
            # First turn: ask for action
            response = self.llm_client.responses.create(
                model=self.llm_model,
                instructions=instructions,
                input=[{"role": "user", "content": user_content}],
                tools=self.osworld_tool,
            )
            
            all_actions = []
            observations = []
            loop = 0
            
            while loop < max_tool_loops:
                loop += 1
                
                # Extract tool calls from this turn
                tool_calls = [item for item in response.output if item.type == "function_call"]
                reasoning = [item for item in response.output if item.type == "reasoning"]
                
                if not tool_calls:
                    # No more tool calls - final answer reached
                    logger.info(f"=== FINAL ANSWER === {response.output_text}")
                    break
                
                logger.info(f"=== TOOL LOOP {loop} === {len(tool_calls)} tool calls")
                next_inputs = []
                
                for call in tool_calls:
                    # Removed method name checking
                    func_args = json.loads(call.arguments or "{}")
                    logger.info(f"Tool call: {call.name}({func_args})")

                    # Some LLMs don't have reasoning content...
                    try:
                        reasoning_content = reasoning[0].content
                    except:
                        reasoning_content = f"Achieving goal: {goal}"
                    
                    # Store the action
                    action_info = {
                        'goal': goal,
                        'tool_name': call.name,
                        'method': func_args.get('method', ''),
                        'params': func_args,
                        'reasoning': reasoning_content,
                        'tool_call_id': call.call_id
                    }
                    all_actions.append(action_info)

                    # Execute the action and get result
                    observation = self.execute_action(action_info, runtime)
                    observations.append(observation)

                    time.sleep(4.0)

                    # Get current state
                    state = self.get_current_state(runtime)

                    step = len(steps) + 1
                    
                    # Save screenshot
                    screenshot_path = self.save_screenshot(
                        state['screenshot'],
                        trajectory_id,
                        step
                    )

                    # Save step data
                    step_data = {
                        'step': step,
                        'timestamp': state['timestamp'],
                        'screenshot': screenshot_path,
                        'ast_xml': state['ast_xml'],  # Keep full AST (not truncated)
                        'simplified_ast': state['simplified_ast'],
                        'cursor_position': state.get('cursor_position', {}),
                        'goal': action_info['goal'],
                        'reasoning': action_info['reasoning'],
                        'action': {
                            'tool_name': action_info['tool_name'],
                            'params': action_info['params']
                        },
                        'observation': observation
                    }
                    steps.append(step_data)
                    observation['Current Screen State'] = state['simplified_ast']
                    observation['Current Cursor Position'] = state['cursor_position']
                    
                    # Truncate observation if too large to prevent context overflow
                    truncated_observation = self._truncate_observation(observation)
                    
                    # Feed result back to model
                    next_inputs.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(truncated_observation)
                    })
                
                if not next_inputs:
                    break
                
                # Continue conversation with previous_response_id
                try:
                    response = self.llm_client.responses.create(
                        model=self.llm_model,
                        previous_response_id=response.id,
                        input=next_inputs,
                        tools=self.osworld_tool,
                        reasoning={"effort": "high"},
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in tool loop continuation: {error_msg}")
                    
                    # Check if it's a max_tokens error (context too large)
                    if 'max_tokens must be at least 1' in error_msg or 'max_tokens' in error_msg.lower():
                        logger.warning("Context is too large - observation exceeds model's context window")
                        logger.warning("Stopping trajectory collection for this step to prevent overflow")
                    
                    # Break the tool loop on any error
                    break
            
            if all_actions:
                return state
            else:
                logger.warning("No actions generated")
                return state
        
        except Exception as e:
            logger.error(f"Error generating action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_action(self, action_info: Dict[str, Any], runtime: OSWorldSingularityRuntime) -> Dict[str, Any]:
        """
        Execute the action selected by LLM.
        Updated for new schema.
        
        Args:
            action_info: Action information from LLM
            runtime: Runtime instance to execute action on
            
        Returns:
            Observation from executing the action
        """
        logger.info(f"Executing action: {action_info['tool_name']}")
        
        # Map shorthand methods to execute_action format
        method = action_info['tool_name']
        params = action_info['params'].copy()

        # Use screen dimensions with fallback defaults
        width = self.screen_width if self.screen_width is not None else self.default_screen_width
        height = self.screen_height if self.screen_height is not None else self.default_screen_height
        if 'x' in params:
            params['x'] = int(params['x'] * width)
        if 'y' in params:
            params['y'] = int(params['y'] * height)
        logger.info(f"Converted normalized coordinates to pixel coordinates: {params}. Screen size: {width}x{height}.")

        try:
            assert method in TOOL_NAME_TO_ACTION_TYPE, f"Unknown tool name: {method}"
            
            action_type = TOOL_NAME_TO_ACTION_TYPE[method]
            action_data = {
            'action_type': action_type,
            'parameters': params
        }
            result = runtime.execute_vm_action(action_data)
            if result.get('status') == 'success':
                result =  {
                    'success': True,
                    'content': result.get('output', 'Action executed successfully'),
                    'exit_code': 0
                }
            else:
                error_msg = result.get('error', result.get('message', 'Unknown error'))
                result =  {
                    'success': False,
                    'content': error_msg,
                    'exit_code': -1
                }
            logger.info(f"Action result: success={result['success']}")
            return result
        
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return {
                'success': False,
                'content': str(e),
                'exit_code': -1
            }
    
    async def collect_trajectory(
        self, 
        trajectory_id: str, 
        runtime: OSWorldSingularityRuntime,
        persona: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Collect a single trajectory using the provided runtime.
        
        Args:
            trajectory_id: Unique identifier for this trajectory
            runtime: Runtime instance to use for this trajectory
            persona: Optional persona context for this trajectory
            
        Returns:
            Trajectory data
        """
        logger.info(f"=" * 80)
        logger.info(f"Collecting trajectory: {trajectory_id}")
        logger.info(f"=" * 80)
        
        if persona:
            logger.info(f"Persona: {persona.get('occupation', 'N/A')} from {persona.get('city', 'N/A')}, {persona.get('state', 'N/A')}")
            logger.info(f"  Age: {persona.get('age', 'N/A')}, Education: {persona.get('education', 'N/A')}")
        
        trajectory = {
            'trajectory_id': trajectory_id,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metadata': {
                'vm_image': self.vm_image_path,
                'llm_model': self.llm_model,
                'screen_size': f"{self.screen_width}x{self.screen_height}",
                'persona': persona  # Include persona in metadata
            }
        }
        
        historical_goals = []  # Track goals separately
        await asyncio.sleep(4.0) # Wait for the UI to update

         # Get current state
        state = self.get_current_state(runtime)
        
        # Save screenshot
        screenshot_path = self.save_screenshot(
            state['screenshot'],
            trajectory_id,
            0
        )
       
        for step in range(self.max_steps_per_trajectory + 1):
            logger.info(f"\nStep {step + 1}/{self.max_steps_per_trajectory}")
            logger.info("-" * 80)
            

            if step == self.max_steps_per_trajectory:
                step_data = {
                    'step': step,
                    'timestamp': state['timestamp'],
                    'screenshot': screenshot_path,
                    'ast_xml': state['ast_xml'],  # Keep full AST (not truncated)
                    'simplified_ast': state['simplified_ast'],
                }
                trajectory['steps'].append(step_data)
                break

            
            # Step 1: Generate sub-goal (with persona context)
            goal = self.generate_goal(state, historical_goals, persona)
            
            if goal is None:
                logger.info("Agent decided to stop or failed to generate goal")
                break
            
            # Add goal to history
            historical_goals.append(goal)
            
            # Step 2: Generate action for the goal
            state = self.generate_action(state, goal, trajectory['steps'], trajectory_id, runtime)
            
            # Check if generate_action failed (returned None)
            if state is None:
                logger.warning("generate_action returned None, stopping trajectory collection")
                break
           
            # Save trajectory incrementally after each step
            trajectory['end_time'] = datetime.now().isoformat()
            trajectory['total_steps'] = len(trajectory['steps'])
            self.save_trajectory(trajectory)
           
            logger.info(f"✓ Step {step + 1} completed and saved")
        
        trajectory['end_time'] = datetime.now().isoformat()
        trajectory['total_steps'] = len(trajectory['steps'])
        
        return trajectory
    
    def save_trajectory(self, trajectory: Dict[str, Any], verbose: bool = False):
        """
        Save trajectory data to JSON file.
        Supports incremental saves - overwrites the file each time with updated data.
        
        Args:
            trajectory: Trajectory data to save
            verbose: If True, log detailed save information (default: False for quieter incremental saves)
        """
        traj_dir = self.output_dir / trajectory['trajectory_id']
        traj_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        traj_file = traj_dir / 'trajectory.json'
        
        try:
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f, indent=2)
            
            if verbose:
                logger.info(f"✓ Trajectory saved: {trajectory['total_steps']} steps to {traj_file}")
            else:
                logger.debug(f"Trajectory checkpoint saved: {trajectory['total_steps']} steps")
        except Exception as e:
            logger.error(f"Failed to save trajectory to {traj_file}: {e}")
    
    def submit_trajectory_job(self, trajectory_idx: int) -> str:
        """
        Submit a trajectory collection job to the queue system.
        
        Args:
            trajectory_idx: Index of this trajectory
            
        Returns:
            Job ID for tracking
        """
        # Create unique IDs
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trajectory_idx:03d}"
        trajectory_id = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trajectory_idx:03d}"
        
        # Sample persona for this trajectory
        persona = self.sample_persona()
        # Sample osworld setup for this trajectory
        # First entry is None so you can skip setup if needed
        osworld_setup = random.choice(self.osworld_setup_dataset)

        # Create job details
        job_details = TrajectoryJobDetails()
        job_details.job_id = job_id
        job_details.trajectory_id = trajectory_id
        job_details.persona = persona
        job_details.osworld_setup = osworld_setup
        job_details.event = threading.Event()
        
        # Store job details
        with self._job_details_lock:
            self._job_details[job_id] = job_details
        
        # Add to init queue
        self.init_queue.put(job_id)
        logger.info(f"Submitted job {job_id} for trajectory {trajectory_id}")
        
        if persona:
            logger.info(f"  Persona: {persona.get('occupation', 'N/A')} from {persona.get('city', 'N/A')}, {persona.get('state', 'N/A')}")
        
        return job_id
    
    async def generate_trajectories(self):
        """Generate multiple trajectories in parallel using queue-based workers."""
        logger.info("=" * 80)
        logger.info("Starting Parallel Synthetic Data Generation")
        logger.info(f"  Parallel workers: {self.max_parallel}")
        logger.info(f"  Total trajectories: {self.max_trajectories}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Start workers
            self.start_workers()
            
            # Submit all trajectory jobs
            job_ids = []
            for i in range(self.max_trajectories):
                job_id = self.submit_trajectory_job(i)
                job_ids.append(job_id)
            
            logger.info(f"\n✓ Submitted {len(job_ids)} trajectory jobs to queue")
            logger.info(f"  Max concurrent runtimes: {self.max_parallel}")
            logger.info(f"  Workers will process jobs sequentially, max {self.max_parallel} runtimes alive at once\n")
            
            # Wait for all jobs to complete
            logger.info("Waiting for all trajectories to complete...")
            for i, job_id in enumerate(job_ids):
                with self._job_details_lock:
                    job_details = self._job_details.get(job_id)
                
                if job_details and job_details.event:
                    job_details.event.wait()  # Block until this job completes
                    
                    # Log progress
                    completed_count = i + 1
                    if completed_count % 10 == 0 or completed_count == len(job_ids):
                        logger.info(f"Progress: {completed_count}/{len(job_ids)} trajectories completed")
            
            # Calculate statistics
            elapsed = time.time() - start_time
            successes = sum(1 for jid in job_ids 
                          if jid in self._job_details and self._job_details[jid].completed)
            failures = len(job_ids) - successes
            
            logger.info("\n" + "=" * 80)
            logger.info("Parallel Synthetic Data Generation Complete")
            logger.info(f"  Successful trajectories: {successes}/{self.max_trajectories}")
            logger.info(f"  Failed trajectories: {failures}")
            logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            logger.info(f"  Avg time per trajectory: {elapsed/max(successes, 1):.1f}s")
            logger.info(f"  Throughput: {successes/(elapsed/60):.2f} trajectories/minute")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info("=" * 80)
        
        finally:
            # Stop workers and cleanup
            self.stop_workers()
            
            # Clear job details
            with self._job_details_lock:
                self._job_details.clear()


async def main():
    """Main function to run synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic computer use trajectories with OSWorld'
    )
    parser.add_argument(
        '--vm-image',
        type=str,
        default='./OS_images/Ubuntu.qcow2',
        help='Path to VM image file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./trajectories',
        help='Output directory for trajectories'
    )
    parser.add_argument(
        '--llm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='LLM API base URL (vLLM server)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='openai/gpt-oss-120b',
        help='LLM model name'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=20,
        help='Maximum steps per trajectory'
    )
    parser.add_argument(
        '--max-trajectories',
        type=int,
        default=10,
        help='Maximum number of trajectories to collect'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=3,
        help='Maximum number of parallel trajectory collectors (default: 3)'
    )
    parser.add_argument(
        '--persona-dataset',
        type=str,
        default='/work/Projects/data/nemotron_data/data',
        help='Path to nemotron persona dataset directory'
    )
    parser.add_argument(
        '--setup-osworld-dataset',
        type=str,
        default='/root/OSWorld/osworld_test_nogdrive.json',
        help='Path to osworld setup dataset directory'
    )
    
    args = parser.parse_args()
    
    # Check VM image exists
    if not Path(args.vm_image).exists():
        print(f"ERROR: VM image not found at {args.vm_image}")
        print("Please provide a valid VM image path with --vm-image")
        return 1
    
    # Create generator
    generator = SyntheticDataGenerator(
        vm_image_path=args.vm_image,
        output_dir=args.output_dir,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
        max_steps_per_trajectory=args.max_steps,
        max_trajectories=args.max_trajectories,
        persona_dataset_path=args.persona_dataset,
        osworld_setup_dataset_path=args.setup_osworld_dataset,
        max_parallel=args.max_parallel,
    )
    
    # Generate trajectories
    await generator.generate_trajectories()
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

