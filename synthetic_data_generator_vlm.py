#!/usr/bin/env python3
"""
Synthetic Data Generator for Computer Use Trajectories (VLM Version)

This script collects trajectories of an AI agent interacting with OSWorld using a VLM:
1. Gets current screen state (screenshot + accessibility tree)
2. VLM imagines a local goal based on screenshot AND AST
3. VLM selects and executes an appropriate action using vision
4. Saves trajectory data (image, AST, simplified AST, goal, action)
5. Repeats for multiple steps

Uses OpenAI-compatible VLM endpoint with vision capabilities.

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
    """Convert tool schema to OpenAI chat completions format."""
    return {
        "type": "function",
        "function": {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"].get("parameters", {}),
        }
    }


def screenshot_to_data_url(screenshot_b64: str) -> str:
    """Convert base64 screenshot to data URL format for VLM."""
    return f"data:image/png;base64,{screenshot_b64}"


def strip_images_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strip image content from all messages except keep text.
    This helps manage context size by removing historical screenshots.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Messages with images removed (text content preserved)
    """
    cleaned_messages = []
    
    for msg in messages:
        new_msg = msg.copy()
        content = msg.get('content')
        
        if isinstance(content, list):
            # Filter out image content, keep only text
            text_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_content.append(item)
                    elif item.get('type') == 'image_url':
                        # ignore image content
                        pass
                        # Replace image with placeholder text
                        # text_content.append({
                        #     'type': 'text',
                        #     'text': '[Previous screenshot omitted to save context]'
                        # })
                else:
                    text_content.append(item)
            
            # If we only have text items, simplify do nothing
            new_msg['content'] = text_content
            # if len(text_content) == 1 and text_content[0].get('type') == 'text':
            #     new_msg['content'] = text_content[0]['text']
            # else:
            #     new_msg['content'] = text_content
        
        cleaned_messages.append(new_msg)
    
    return cleaned_messages


class TrajectoryJobDetails:
    """Details for a single trajectory collection job."""
    def __init__(self):
        self.job_id: str = ''
        self.trajectory_id: str = ''
        self.persona: Optional[Dict[str, Any]] = None
        self.osworld_setup: Optional[Dict[str, Any]] = None
        self.runtime: Optional[OSWorldSingularityRuntime] = None
        self.trajectory_data: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.event: Optional[threading.Event] = None
        self.completed: bool = False


class SyntheticDataGeneratorVLM:
    """Generates synthetic computer use trajectories using OSWorld and VLM (Vision Language Model)."""
    
    def __init__(
        self,
        vm_image_path: str,
        output_dir: str = './trajectories',
        vlm_base_url: str = 'http://localhost:8000/v1',
        vlm_model: str = 'internvl3_5-30b',
        max_steps_per_trajectory: int = 20,
        max_trajectories: int = 10,
        persona_dataset_path: Optional[str] = '/work/Projects/data/nemotron_data/data',
        osworld_setup_dataset_path: Optional[str] = '/root/OSWorld/osworld_test_nogdrive.json',
        max_parallel: int = 3,
    ):
        """
        Initialize the synthetic data generator with VLM support.
        
        Args:
            vm_image_path: Path to the VM image file
            output_dir: Directory to save trajectories
            vlm_base_url: Base URL for VLM API (OpenAI-compatible)
            vlm_model: VLM model name (e.g., internvl3_5-30b)
            max_steps_per_trajectory: Maximum steps per trajectory
            max_trajectories: Maximum number of trajectories to collect
            persona_dataset_path: Path to nemotron persona dataset (parquet files)
            max_parallel: Maximum number of parallel trajectory collectors
        """
        self.vm_image_path = vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in vm_image_path else 'windows'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vlm_base_url = vlm_base_url
        self.vlm_model = vlm_model
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.max_trajectories = max_trajectories
        self.max_parallel = max_parallel
        
        # Initialize VLM client (OpenAI-compatible)
        self.vlm_client = OpenAI(
            base_url=vlm_base_url,
            api_key='EMPTY',  # Most VLM servers ignore this
        )
        
        # Get OSWorld tool definitions for tool calling
        self.osworld_tools = []
        # Remove finish, wait, and fail tools (last 3)
        for tool in OSWORLD_TOOLS[:-3]:
            self.osworld_tools.append(fix_tool_schema(tool))
        
        # Queue-based architecture
        self.init_queue: queue.Queue[str] = queue.Queue()
        self.collect_queue: queue.Queue[str] = queue.Queue()
        self._init_workers: List[Optional[asyncio.AbstractEventLoop]] = []
        self._collect_workers: List[Optional[asyncio.AbstractEventLoop]] = []
        self._job_details: Dict[str, TrajectoryJobDetails] = {}
        self._job_details_lock = threading.RLock()
        self._server_running: bool = False
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Concurrency control
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
        self.persona_dfs = []
        self.persona_df_weights = []
        self.persona_dataset_path = persona_dataset_path
        if persona_dataset_path and os.path.exists(persona_dataset_path):
            self._load_persona_dataset()
        else:
            logger.warning(f"Persona dataset path not found: {persona_dataset_path}")
            logger.warning("  Continuing without persona-based goal generation")

        # Load OSWorld setup dataset
        self.osworld_setup_dataset = [None]
        self.osworld_setup_dataset_path = osworld_setup_dataset_path
        if osworld_setup_dataset_path and os.path.exists(osworld_setup_dataset_path):
            self._load_osworld_setup_dataset()
        else:
            logger.warning(f"OSWorld setup dataset path not found: {osworld_setup_dataset_path}")
            logger.warning("  Continuing without OSWorld setup")
        
        logger.info(f"Initialized SyntheticDataGeneratorVLM")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  VLM: {vlm_model} @ {vlm_base_url}")
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
            parquet_files = list(Path(self.persona_dataset_path).glob('train-*.parquet'))
            
            if not parquet_files:
                logger.warning(f"No parquet files found in {self.persona_dataset_path}")
                return
            
            logger.info(f"  Found {len(parquet_files)} parquet files")
            
            self.persona_dfs = []
            self.persona_df_weights = []
            total_records = 0
            
            for pf in parquet_files:
                df = pd.read_parquet(pf, memory_map=True)
                self.persona_dfs.append(df)
                self.persona_df_weights.append(len(df))
                total_records += len(df)
                logger.info(f"    Loaded {pf.name}: {len(df):,} records")
            
            logger.info(f"  ✓ Loaded {total_records:,} total persona records from {len(parquet_files)} files")
            
        except Exception as e:
            logger.error(f"Error loading persona dataset: {e}")
            self.persona_dfs = []
            self.persona_df_weights = []

    def _load_osworld_setup_dataset(self):
        """Load the osworld setup dataset from json files."""
        logger.info(f"Loading osworld setup dataset from {self.osworld_setup_dataset_path}...")
        
        try:
            self.osworld_setup_dataset = [None]
            with open(self.osworld_setup_dataset_path, 'r') as f:
                for line in f.readlines():
                    self.osworld_setup_dataset.append(json.loads(line))
            logger.info(f"  ✓ Loaded {len(self.osworld_setup_dataset)} total osworld setup records")
            
        except Exception as e:
            logger.error(f"Error loading osworld setup dataset: {e}")
            self.osworld_setup_dataset = [None]
    
    def sample_persona(self) -> Optional[Dict[str, Any]]:
        """Sample a random persona from the dataset."""
        if not self.persona_dfs or sum(self.persona_df_weights) == 0:
            return None
        
        selected_df = random.choices(self.persona_dfs, weights=self.persona_df_weights, k=1)[0]
        persona_record = selected_df.sample(n=1).iloc[0].to_dict()
        
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
        
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_parallel * 2
        )
        
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
        
        for _ in range(self.max_parallel):
            try:
                self.init_queue.put_nowait('__STOP__')
                self.collect_queue.put_nowait('__STOP__')
            except Exception as e:
                logger.warning(f"Error sending stop signal: {e}")
        
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logger.info("✓ Workers stopped")
    
    def _cleanup_job_runtime(self, runtime: OSWorldSingularityRuntime, job_id: str):
        """Cleanup runtime in background thread."""
        def close():
            try:
                runtime.close()
                if hasattr(runtime, 'event_stream') and runtime.event_stream:
                    try:
                        runtime.event_stream.close()
                    except Exception as e:
                        logger.warning(f'Error closing event stream for job {job_id}: {e}')
                time.sleep(0.1)
            except Exception as e:
                logger.error(f'Error cleaning up runtime for job {job_id}: {e}')
        
        t = threading.Thread(target=close, daemon=True)
        t.start()
    
    async def _init_worker(self, worker_id: int):
        """Init worker: initializes runtimes for trajectory jobs."""
        logger.info(f"[init-worker-{worker_id}] Started")
        
        while True:
            job_id = await asyncio.to_thread(self.init_queue.get)
            
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
            
            logger.info(f"[init-worker-{worker_id}] Waiting for runtime slot...")
            await asyncio.to_thread(self._runtime_semaphore.acquire)
            
            with self._active_runtime_lock:
                self._active_runtime_count += 1
            
            logger.info(f"[init-worker-{worker_id}] Runtime slot acquired ({self._active_runtime_count}/{self.max_parallel})")
            
            try:
                config = OpenHandsConfig()
                config.runtime = 'osworld'
                config.sandbox.base_container_image = 'ubuntu:24.04'
                config.sandbox.run_as_fakeroot = True
                config.sandbox.runtime_container_image = None
                
                file_store = get_file_store('local', f'/tmp/synthetic_data_gen_{job_id}')
                event_stream = EventStream(sid=job_id, file_store=file_store)
                
                if not os.path.exists(self.vm_image_path):
                    raise RuntimeError(f"VM image not found: {self.vm_image_path}")
                
                logger.info(f"[init-worker-{worker_id}] Creating runtime for {job_id}")
                
                runtime = OSWorldSingularityRuntime(
                    config=config,
                    event_stream=event_stream,
                    sid=job_id,
                    os_type=self.os_type,
                    vm_image_path=self.vm_image_path,
                    attach_to_existing=False,
                )
                
                await runtime.connect()
                logger.info(f"[init-worker-{worker_id}] ✓ Runtime initialized for {job_id}")

                if job_details.osworld_setup and self.os_type == 'linux':
                    logger.info(f"[init-worker-{worker_id}] Setting up OSWorld...")
                    setup_controller = SetupController(
                        vm_ip="127.0.0.1",
                        server_port=runtime._vm_server_port,
                        chromium_port=runtime._chromium_port,
                        cache_dir="/tmp/osworld_example",
                        client_password="password",
                        runtime=runtime  
                    )
                    await setup_controller.setup(job_details.osworld_setup['config'])
                    logger.info(f"[init-worker-{worker_id}] ✓ OSWorld setup completed")
                
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].runtime = runtime
                        self.collect_queue.put(job_id)
                
            except Exception as e:
                logger.error(f"[init-worker-{worker_id}] Failed to init runtime for {job_id}: {e}")
                with self._job_details_lock:
                    if job_id in self._job_details:
                        self._job_details[job_id].error = str(e)
                        self._job_details[job_id].event.set()
                
                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1
            
            finally:
                self.init_queue.task_done()
    
    async def _collect_worker(self, worker_id: int):
        """Collect worker: collects trajectories using initialized runtimes."""
        logger.info(f"[collect-worker-{worker_id}] Started")
        
        while True:
            job_id = await asyncio.to_thread(self.collect_queue.get)
            
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
                trajectory_data = await self.collect_trajectory(
                    job_details.trajectory_id,
                    job_details.runtime,
                    job_details.persona
                )
                
                self.save_trajectory(trajectory_data, verbose=True)
                
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
                if job_details.runtime:
                    logger.info(f"[collect-worker-{worker_id}] Cleaning up runtime for {job_id}")
                    self._cleanup_job_runtime(job_details.runtime, job_id)
                    with self._job_details_lock:
                        if job_id in self._job_details:
                            self._job_details[job_id].runtime = None
                
                self._runtime_semaphore.release()
                with self._active_runtime_lock:
                    self._active_runtime_count -= 1
                logger.info(f"[collect-worker-{worker_id}] Runtime slot released ({self._active_runtime_count}/{self.max_parallel})")
                
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
            except Exception as e:
                logger.warning(f'Error closing event loop for collect worker {worker_id}: {e}')
    
    def get_current_state(self, runtime: OSWorldSingularityRuntime) -> Dict[str, Any]:
        """
        Get the current screen state including screenshot and accessibility tree.
        
        Returns:
            Dictionary with screenshot (base64), AST, simplified AST, and cursor position
        """
        logger.info("Getting current screen state...")
        
        # Get screenshot
        screenshot_bytes = runtime.get_vm_screenshot()
        if screenshot_bytes:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        else:
            screenshot_b64 = ''
        
        # Get cursor position
        cursor_x, cursor_y = 0, 0
        try:
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
        except Exception as e:
            logger.warning(f"Could not get cursor position: {e}")
        
        # Get accessibility tree
        action = OSWorldInteractiveAction(
            method='get_accessibility_tree',
            params={},
            thought='Getting UI accessibility tree'
        )
        ast_obs = runtime.run_action(action)
        
        try:
            ast_data = json.loads(ast_obs.content)
            ast_xml = ast_data.get('AT', '')
        except:
            ast_xml = ast_obs.content
        
        # Simplify AST
        raw_ast = None
        if self.os_type == 'windows':
            simplified_ast, (self.screen_width, self.screen_height) = simplify_windows_accessibility_tree(ast_xml)
            raw_ast = ast_xml
        else:
            simplified_ast = ast_xml[0]
            raw_ast = ast_xml[0]
            self.screen_width = None
            self.screen_height = None
        
        # Normalize cursor position
        width = self.screen_width if self.screen_width is not None else self.default_screen_width
        height = self.screen_height if self.screen_height is not None else self.default_screen_height
        
        normalized_cursor_x = cursor_x / width if width > 0 else 0
        normalized_cursor_y = cursor_y / height if height > 0 else 0
        
        return {
            'screenshot': screenshot_b64,
            'ast_xml': raw_ast,
            'simplified_ast': simplified_ast,
            'cursor_position': {'x': round(normalized_cursor_x, 3), 'y': round(normalized_cursor_y, 3)},
            'timestamp': time.time()
        }
    
    def _build_vlm_message_with_image(
        self,
        text_content: str,
        screenshot_b64: str
    ) -> List[Dict[str, Any]]:
        """
        Build a VLM message that includes both text and image.
        
        Args:
            text_content: The text part of the message
            screenshot_b64: Base64 encoded screenshot
            
        Returns:
            List of content items for the message
        """
        content = [
            {"type": "text", "text": text_content},
        ]
        
        if screenshot_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": screenshot_to_data_url(screenshot_b64)}
            })
        
        return content
    
    def save_screenshot(self, screenshot_content: str, trajectory_id: str, step: int) -> str:
        """Save screenshot to file."""
        traj_dir = self.output_dir / trajectory_id
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        screenshot_path = traj_dir / f"screenshot_step_{step:03d}.png"
        
        try:
            screenshot_b64 = screenshot_content
            if screenshot_b64.startswith('base64:'):
                screenshot_b64 = screenshot_b64[7:]
            
            screenshot_data = base64.b64decode(screenshot_b64)
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_data)
        except Exception as e:
            if os.path.exists(screenshot_content):
                import shutil
                shutil.copy(screenshot_content, screenshot_path)
            else:
                logger.warning(f"Could not decode screenshot: {e}")
                from PIL import Image
                img = Image.new('RGB', (800, 600), color='gray')
                img.save(screenshot_path)
        
        return str(screenshot_path.relative_to(self.output_dir))
    
    def generate_goal_vlm(
        self,
        state: Dict[str, Any],
        historical_goals: List[str],
        persona: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Use VLM to generate a sub-goal based on screenshot AND AST.
        
        Args:
            state: Current screen state with screenshot and simplified AST
            historical_goals: List of previously generated goals
            persona: Optional persona information
            
        Returns:
            Goal string, or None if VLM decides to stop
        """
        logger.info("Generating sub-goal with VLM (screenshot + AST)...")
        
        # Format historical goals
        goals_history = ""
        if historical_goals:
            goals_history = "\n\n**Previous goals you've pursued:**\n"
            for i, goal in enumerate(historical_goals): 
                goals_history += f"{i+1}. {goal}\n"
            goals_history += "\nTry to finish the previous goal. e.g. if you clicked on URL field, your next goal can be typing the URL\n"
        
        # Prepare persona context
        persona_context = ""
        if persona:
            persona_context = f"\n\n**Your Persona Context:**\n"
            
            if persona.get('occupation'):
                occupation = persona['occupation'].replace('_', ' ').title()
                persona_context += f"- Occupation: {occupation}"
                if persona.get('age'):
                    persona_context += f" (age {persona['age']})"
                if persona.get('city') and persona.get('state'):
                    persona_context += f" from {persona['city']}, {persona['state']}"
                persona_context += "\n"
            
            if persona.get('interests_list'):
                try:
                    import ast
                    interests = ast.literal_eval(persona['interests_list'])
                    if interests and len(interests) > 0:
                        sample_interests = random.sample(interests, min(5, len(interests)))
                        persona_context += f"- Interests: {', '.join(sample_interests)}\n"
                except:
                    pass
            
            if persona.get('professional'):
                prof_text = persona['professional'][:200].strip()
                if len(persona['professional']) > 200:
                    last_period = prof_text.rfind('.')
                    if last_period > 100:
                        prof_text = prof_text[:last_period+1]
                persona_context += f"- Work style: {prof_text}\n"
            
            persona_context += "\nGenerate goals that align with this persona's background and interests.\n"
        
        cursor_info = state.get('cursor_position', {})
        cursor_text = f"**Current Cursor Position:** ({cursor_info.get('x', 'unknown')}, {cursor_info.get('y', 'unknown')})\n\n"
        
        # Build user message text
        user_text = f"""Look at the current screenshot and the UI accessibility tree below.

{cursor_text}{persona_context}

**UI Accessibility Tree (for precise element coordinates):**
{state['simplified_ast']}

{goals_history}

**Guidelines:**
- Don't ask clarification questions - just generate a simple goal
- Choose realistic, achievable goals from visible UI elements
- Goals should be specific and actionable (e.g., "Type 'news' in search box", "Open Google Chrome")
- Goals must be atomic - ONE action at a time
- Be curious and explore different parts of the system
- If the same goal appears multiple times, try a different approach
- Generate coherent goal sequences (e.g., if browser is open, search for something)
- Goals must be achievable with current screen state
{('- If persona context is provided, generate goals aligned with their interests' if persona else '')}

What specific sub-goal would you like to achieve next?"""
        system_prompt = """You are an AI agent that rigorously follows this response protocol:
- You are an AI agent exploring a Ubuntu desktop environment. 
- You will be shown a screenshot of the current screen along with the UI accessibility tree. 
- Your task is to imagine ONE reasonable sub-goal you could achieve based on what you see. 
- Respond with a single, specific goal.

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone goal and not reference the thinking section.
"""

 
        # Build message with image
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": self._build_vlm_message_with_image(user_text, state['screenshot']),
            }
        ]
        
        try:
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                max_tokens=256,
                temperature=0.3,
            )
            
            goal = response.choices[0].message.content.strip() if response.choices else ""
            
            if not goal:
                logger.warning("VLM returned empty goal")
                return None
            
            logger.info(f"Generated goal: {goal}")
            return goal
        
        except Exception as e:
            logger.error(f"Error generating goal with VLM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_action_vlm(
        self,
        state: Dict[str, Any],
        goal: str,
        steps: List[Dict[str, Any]],
        trajectory_id: str,
        runtime: OSWorldSingularityRuntime,
        max_tool_loops: int = 10
    ) -> Optional[tuple]:
        """
        Use VLM to select an action based on screenshot, AST, and goal.
        Uses tool calling loop with vision.
        
        Args:
            state: Current screen state with screenshot and simplified AST
            goal: The sub-goal to achieve
            steps: List of step data
            trajectory_id: Trajectory identifier
            runtime: Runtime instance
            max_tool_loops: Maximum tool call iterations
            
        Returns:
            Tuple of (updated_state, final_message), or None if failed
        """
        logger.info(f"Generating action with VLM for goal: {goal}")
        final_message = ""
        
        cursor_info = state.get('cursor_position', {})
        cursor_text = f"**Current Cursor Position:** ({cursor_info.get('x', 'unknown')}, {cursor_info.get('y', 'unknown')})\n\n"
        
        # Build initial user message with image and AST
        user_text = f"""Look at the current screenshot and the UI accessibility tree below.

{cursor_text}

**UI Accessibility Tree (for precise element coordinates):**
{state['simplified_ast']}

**Goal:** {goal}

Select the appropriate action using the available tools. Use coordinates from the AST for precise targeting."""
        system_prompt = """You are an AI agent that rigorously follows this response protocol:
- You are an AI agent controlling a Ubuntu desktop environment. 
- You will be shown screenshots along with the UI accessibility tree.
- Use the provided tools to interact with the desktop.
- Use exact coordinates from the accessibility tree for clicks and interactions.

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone tool call action and not reference the thinking section.
"""

     
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": self._build_vlm_message_with_image(user_text, state['screenshot']),
            }
        ]
        
        try:
            # First VLM call
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                tools=self.osworld_tools,
                max_tokens=512,
                temperature=0.2,
            )
            
            all_actions = []
            loop = 0
            
            while loop < max_tool_loops:
                loop += 1
                
                msg = response.choices[0].message
                
                if not msg.tool_calls:
                    # No more tool calls - final answer reached
                    logger.info(f"=== FINAL ANSWER === {msg.content}")
                    final_message = f"     >>> final message for this goal: {msg.content}"
                    break
                
                logger.info(f"=== VLM TOOL LOOP {loop} === {len(msg.tool_calls)} tool calls")
                
                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in msg.tool_calls
                    ]
                })
                
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments or "{}")
                    logger.info(f"Tool call: {func_name}({func_args})")
                    
                    # Store the action
                    action_info = {
                        'goal': goal,
                        'tool_name': func_name,
                        'method': func_name,
                        'params': func_args,
                        'reasoning': msg.content or f"Achieving goal: {goal}",
                        'tool_call_id': tool_call.id
                    }
                    all_actions.append(action_info)

                    # Execute the action
                    observation = self.execute_action(action_info, runtime)

                    time.sleep(4.0)

                    # Get updated state with new screenshot
                    state = self.get_current_state(runtime)

                    step = len(steps) + 1
                    
                    # Save screenshot
                    screenshot_path = self.save_screenshot(
                        state['screenshot'],
                        trajectory_id,
                        step
                    )
                    
                    logger.info(f"Action reasoning: {action_info['reasoning']}")

                    # Save step data
                    step_data = {
                        'step': step,
                        'timestamp': state['timestamp'],
                        'screenshot': screenshot_path,
                        'ast_xml': state['ast_xml'],
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
                    
                    # Build tool result with new screenshot
                    observation_text = json.dumps({
                        **observation,
                        'current_cursor_position': state['cursor_position']
                    })
                    
                    # Add tool result message with NEW screenshot
                    tool_result_text = f"""Action result: {observation_text}

Here is the updated screen after the action. The UI Accessibility Tree is now:
{state['simplified_ast']}

Continue with the goal or indicate if it's achieved."""
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._build_vlm_message_with_image(tool_result_text, state['screenshot']),
                    })
                    break
                
                # Continue conversation
                # Strip old images from history to manage context size
                # Only the latest tool result will have the new screenshot
                messages = strip_images_from_messages(messages[:-1]) + [messages[-1]]
                
                try:
                    response = self.vlm_client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        tools=self.osworld_tools,
                        tool_choice="auto",
                        max_tokens=512,
                        temperature=0.2,
                    )
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error in VLM tool loop continuation: {error_msg}")
                    
                    if 'max_tokens' in error_msg.lower():
                        logger.warning("Context is too large - stopping trajectory")
                    
                    break
            
            if all_actions:
                return state, final_message
            else:
                logger.warning("No actions generated by VLM")
                return state, final_message
        
        except Exception as e:
            logger.error(f"Error generating action with VLM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def execute_action(self, action_info: Dict[str, Any], runtime: OSWorldSingularityRuntime) -> Dict[str, Any]:
        """Execute the action selected by VLM."""
        logger.info(f"Executing action: {action_info['tool_name']}")
        
        method = action_info['tool_name']
        params = action_info['params'].copy()

        # Use screen dimensions with fallback defaults
        width = self.screen_width if self.screen_width is not None else self.default_screen_width
        height = self.screen_height if self.screen_height is not None else self.default_screen_height
        
        if 'x' in params:
            params['x'] = int(params['x'] * width)
        if 'y' in params:
            params['y'] = int(params['y'] * height)
        logger.info(f"Converted coordinates: {params}. Screen size: {width}x{height}.")

        try:
            assert method in TOOL_NAME_TO_ACTION_TYPE, f"Unknown tool name: {method}"
            
            action_type = TOOL_NAME_TO_ACTION_TYPE[method]
            action_data = {
                'action_type': action_type,
                'parameters': params
            }
            result = runtime.execute_vm_action(action_data)
            
            if result.get('status') == 'success':
                return {
                    'success': True,
                    'content': result.get('output', 'Action executed successfully'),
                    'exit_code': 0
                }
            else:
                error_msg = result.get('error', result.get('message', 'Unknown error'))
                return {
                    'success': False,
                    'content': error_msg,
                    'exit_code': -1
                }
        
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
        """Collect a single trajectory using VLM."""
        logger.info(f"=" * 80)
        logger.info(f"Collecting VLM trajectory: {trajectory_id}")
        logger.info(f"=" * 80)
        
        if persona:
            logger.info(f"Persona: {persona.get('occupation', 'N/A')} from {persona.get('city', 'N/A')}, {persona.get('state', 'N/A')}")
        
        trajectory = {
            'trajectory_id': trajectory_id,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metadata': {
                'vm_image': self.vm_image_path,
                'vlm_model': self.vlm_model,
                'screen_size': f"{self.screen_width}x{self.screen_height}",
                'persona': persona,
                'generator_type': 'vlm'  # Mark as VLM-generated
            }
        }
        
        historical_goals = []
        await asyncio.sleep(4.0)

        # Get initial state
        state = self.get_current_state(runtime)
        
        # Save initial screenshot
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
                    'ast_xml': state['ast_xml'],
                    'simplified_ast': state['simplified_ast'],
                }
                trajectory['steps'].append(step_data)
                break
            
            # Generate goal using VLM (with screenshot + AST)
            goal = self.generate_goal_vlm(state, historical_goals, persona)
            
            if goal is None:
                logger.info("VLM decided to stop or failed to generate goal")
                break
            
            # Generate action using VLM (with screenshot + AST + goal)
            result = self.generate_action_vlm(state, goal, trajectory['steps'], trajectory_id, runtime)

            if result is None:
                logger.warning("generate_action_vlm returned None, stopping")
                break
            
            state, final_message = result
            
            # Add goal to history
            historical_goals.append(f"{goal}\n{final_message}")
           
            # Save trajectory incrementally
            trajectory['end_time'] = datetime.now().isoformat()
            trajectory['total_steps'] = len(trajectory['steps'])
            self.save_trajectory(trajectory)
           
            logger.info(f"✓ Step {step + 1} completed and saved")
        
        trajectory['end_time'] = datetime.now().isoformat()
        trajectory['total_steps'] = len(trajectory['steps'])
        
        return trajectory
    
    def save_trajectory(self, trajectory: Dict[str, Any], verbose: bool = False):
        """Save trajectory data to JSON file."""
        traj_dir = self.output_dir / trajectory['trajectory_id']
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_file = traj_dir / 'trajectory.json'
        
        try:
            with open(traj_file, 'w') as f:
                json.dump(trajectory, f, indent=2)
            
            if verbose:
                logger.info(f"✓ Trajectory saved: {trajectory['total_steps']} steps to {traj_file}")
        except Exception as e:
            logger.error(f"Failed to save trajectory to {traj_file}: {e}")
    
    def submit_trajectory_job(self, trajectory_idx: int) -> str:
        """Submit a trajectory collection job to the queue system."""
        job_id = f"vlm_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trajectory_idx:03d}"
        trajectory_id = f"vlm_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trajectory_idx:03d}"
        
        persona = self.sample_persona()
        osworld_setup = random.choice(self.osworld_setup_dataset)

        job_details = TrajectoryJobDetails()
        job_details.job_id = job_id
        job_details.trajectory_id = trajectory_id
        job_details.persona = persona
        job_details.osworld_setup = osworld_setup
        job_details.event = threading.Event()
        
        with self._job_details_lock:
            self._job_details[job_id] = job_details
        
        self.init_queue.put(job_id)
        logger.info(f"Submitted VLM job {job_id} for trajectory {trajectory_id}")
        
        if persona:
            logger.info(f"  Persona: {persona.get('occupation', 'N/A')} from {persona.get('city', 'N/A')}")
        
        return job_id
    
    async def generate_trajectories(self):
        """Generate multiple trajectories in parallel using VLM."""
        logger.info("=" * 80)
        logger.info("Starting Parallel VLM Synthetic Data Generation")
        logger.info(f"  VLM Model: {self.vlm_model}")
        logger.info(f"  Parallel workers: {self.max_parallel}")
        logger.info(f"  Total trajectories: {self.max_trajectories}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            self.start_workers()
            
            job_ids = []
            for i in range(self.max_trajectories):
                job_id = self.submit_trajectory_job(i)
                job_ids.append(job_id)
            
            logger.info(f"\n✓ Submitted {len(job_ids)} VLM trajectory jobs to queue")
            
            # Wait for all jobs to complete
            logger.info("Waiting for all trajectories to complete...")
            for i, job_id in enumerate(job_ids):
                with self._job_details_lock:
                    job_details = self._job_details.get(job_id)
                
                if job_details and job_details.event:
                    job_details.event.wait()
                    
                    completed_count = i + 1
                    if completed_count % 10 == 0 or completed_count == len(job_ids):
                        logger.info(f"Progress: {completed_count}/{len(job_ids)} trajectories completed")
            
            elapsed = time.time() - start_time
            successes = sum(1 for jid in job_ids 
                          if jid in self._job_details and self._job_details[jid].completed)
            failures = len(job_ids) - successes
            
            logger.info("\n" + "=" * 80)
            logger.info("VLM Synthetic Data Generation Complete")
            logger.info(f"  Successful trajectories: {successes}/{self.max_trajectories}")
            logger.info(f"  Failed trajectories: {failures}")
            logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            logger.info(f"  Avg time per trajectory: {elapsed/max(successes, 1):.1f}s")
            logger.info(f"  Throughput: {successes/(elapsed/60):.2f} trajectories/minute")
            logger.info(f"  Output directory: {self.output_dir}")
            logger.info("=" * 80)
        
        finally:
            self.stop_workers()
            
            with self._job_details_lock:
                self._job_details.clear()


async def main():
    """Main function to run VLM synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic computer use trajectories with VLM (Vision Language Model)'
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
        default='./trajectories_vlm',
        help='Output directory for trajectories'
    )
    parser.add_argument(
        '--vlm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='VLM API base URL (OpenAI-compatible endpoint)'
    )
    parser.add_argument(
        '--vlm-model',
        type=str,
        default='internvl3_5-30b',
        help='VLM model name (e.g., internvl3_5-30b)'
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
        help='Maximum number of parallel trajectory collectors'
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
    generator = SyntheticDataGeneratorVLM(
        vm_image_path=args.vm_image,
        output_dir=args.output_dir,
        vlm_base_url=args.vlm_url,
        vlm_model=args.vlm_model,
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

