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
import sys
import time
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
from openhands.utils.ast_process import simplify_accessibility_tree
from openhands.agenthub.codeact_agent.tools.osworld import get_osworld_tool_vllm
import time


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
        """
        self.vm_image_path = vm_image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.max_trajectories = max_trajectories
        
        # Initialize LLM client
        self.llm_client = OpenAI(
            base_url=llm_base_url,
            api_key='EMPTY',  # vLLM ignores this by default
        )
        
        # Get OSWorld tool definition for vLLM
        self.osworld_tool = get_osworld_tool_vllm()
        
        # Runtime will be initialized in async context
        self.runtime: Optional[OSWorldSingularityRuntime] = None
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Load persona dataset
        self.persona_df = None
        self.persona_dataset_path = persona_dataset_path
        if persona_dataset_path and os.path.exists(persona_dataset_path):
            self._load_persona_dataset()
        else:
            logger.warning(f"Persona dataset path not found: {persona_dataset_path}")
            logger.warning("  Continuing without persona-based goal generation")
        
        logger.info(f"Initialized SyntheticDataGenerator")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  LLM: {llm_model} @ {llm_base_url}")
        logger.info(f"  Max steps per trajectory: {max_steps_per_trajectory}")
        logger.info(f"  Max trajectories: {max_trajectories}")
        if self.persona_df is not None:
            logger.info(f"  Personas loaded: {len(self.persona_df):,} records")
    
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
            
            # Load first file for efficiency (contains ~90k personas)
            # Can load all files if needed: pd.concat([pd.read_parquet(f) for f in parquet_files])
            self.persona_df = pd.read_parquet(parquet_files[0])
            
            logger.info(f"  ✓ Loaded {len(self.persona_df):,} persona records")
            logger.info(f"  Fields: {', '.join(self.persona_df.columns[:8])}...")
            
        except Exception as e:
            logger.error(f"Error loading persona dataset: {e}")
            self.persona_df = None
    
    def sample_persona(self) -> Optional[Dict[str, Any]]:
        """
        Sample a random persona from the dataset.
        
        Returns:
            Dictionary containing persona information, or None if dataset not loaded
        """
        if self.persona_df is None or len(self.persona_df) == 0:
            return None
        
        # Sample random persona
        persona_record = self.persona_df.sample(n=1).iloc[0].to_dict()
        
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
    
    async def initialize_runtime(self):
        """Initialize OSWorld runtime and connect to VM."""
        logger.info("Initializing OSWorld runtime...")
        
        # Create configuration
        config = OpenHandsConfig()
        config.runtime = 'osworld'
        config.sandbox.base_container_image = 'ubuntu:24.04'
        config.sandbox.run_as_fakeroot = True
        
        # Create event stream
        file_store = get_file_store('local', '/tmp/synthetic_data_gen')
        event_stream = EventStream(sid='synthetic-data-gen', file_store=file_store)
        
        # Create runtime
        self.runtime = OSWorldSingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid='synthetic-data-gen',
            os_type='linux',
            vm_image_path=self.vm_image_path,
            attach_to_existing=False,
        )
        
        # Connect to VM
        logger.info("Connecting to VM (this may take 1-2 minutes)...")
        await self.runtime.connect()
        logger.info(f"✓ Runtime connected! VM URL: {self.runtime.osworld_vm_url}")
        
        # Get screen size
        try:
            action = OSWorldInteractiveAction(
                method='get_vm_screen_size',
                params={},
                thought='Getting screen dimensions'
            )
            observation = self.runtime.run_action(action)
            if 'width' in observation.content and 'height' in observation.content:
                screen_info = json.loads(observation.content)
                self.screen_width = screen_info['width']
                self.screen_height = screen_info['height']
                logger.info(f"Screen size: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            logger.warning(f"Could not get screen size, using default: {e}")
    
    async def cleanup_runtime(self):
        """Clean up runtime resources."""
        if self.runtime:
            logger.info("Cleaning up runtime...")
            try:
                # close() might not be async
                if hasattr(self.runtime, 'close'):
                    result = self.runtime.close()
                    # If it returns a coroutine, await it
                    if hasattr(result, '__await__'):
                        await result
                logger.info("✓ Runtime closed")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current screen state including screenshot, accessibility tree, and cursor position.
        
        Returns:
            Dictionary with screenshot, AST, simplified AST, and cursor position
        """
        logger.info("Getting current screen state...")
        
        # Get screenshot directly (not as action)
        screenshot_bytes = self.runtime.get_vm_screenshot()
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
            cursor_obs = self.runtime.run_action(action)
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
        ast_obs = self.runtime.run_action(action)
        
        # Parse AST (it's returned as JSON with 'AT' key containing XML)
        try:
            ast_data = json.loads(ast_obs.content)
            ast_xml = ast_data.get('AT', '')
        except:
            ast_xml = ast_obs.content
        
        # Simplify AST for LLM using ast_process simplifier
        # This returns clean XML with center coordinates and bounding boxes
        simplified_ast = simplify_accessibility_tree(ast_xml)
        
        return {
            'screenshot': screenshot_b64,
            'ast_xml': ast_xml,
            'simplified_ast': simplified_ast,
            'cursor_position': {'x': cursor_x, 'y': cursor_y},
            'timestamp': time.time()
        }
    
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
        max_tool_loops: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Use LLM to select an action based on the goal and current state.
        Uses tool calling loop until final answer (following client_test.py pattern).
        
        Args:
            state: Current screen state with simplified AST
            goal: The sub-goal to achieve
            conversation_history: Previous conversation messages
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
1. Type text: execute_action with TYPING action_type
2. Click elements: execute_action with CLICK action_type  
3. Press keys: execute_action with PRESS action_type

Use the osworld tool to interact. Look at the screen state and cursor position, select the appropriate action.
Use exact coordinates from the simplified AST.

Example for typing:
  osworld(method="execute_action", params={{"action": {{"action_type": "TYPING", "parameters": {{"text": "ubuntu.com"}}}}}})

Example for clicking:
  osworld(method="execute_action", params={{"action": {{"action_type": "CLICK", "parameters": {{"x": 100, "y": 200, "button": "left"}}}}}})"""

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
                tools=[self.osworld_tool],
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
                    # vLLM tool format has name at top level
                    if call.name != self.osworld_tool['name']:
                        logger.warning(f"Ignoring unknown tool: {call.name}")
                        next_inputs.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps({
                                "success": False,
                                "content": f"Unknown tool: {call.name}",
                                "exit_code": -1
                            })
                        })
                        continue
                    
                    func_args = json.loads(call.arguments or "{}")
                    logger.info(f"Tool call: {call.name}({func_args})")
                    
                    # Store the action
                    action_info = {
                        'goal': goal,
                        'tool_name': call.name,
                        'method': func_args.get('method', ''),
                        'params': func_args.get('params', {}),
                        'reasoning': reasoning[0].content if reasoning else f"Achieving goal: {goal}",
                        'tool_call_id': call.call_id
                    }
                    all_actions.append(action_info)
                    
                    # Execute the action and get result
                    observation = self.execute_action(action_info)
                    observations.append(observation)

                    time.sleep(4.0)

                    # Get current state
                    state = self.get_current_state()

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
                            'method': action_info['method'],
                            'params': action_info['params']
                        },
                        'observation': observation
                    }
                    steps.append(step_data)
                    observation['Current Screen State'] = state['simplified_ast']
                    observation['Current Cursor Position'] = state['cursor_position']
                    
                    # Feed result back to model
                    next_inputs.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(observation)
                    })
                
                if not next_inputs:
                    break
                
                # Continue conversation with previous_response_id
                try:
                    response = self.llm_client.responses.create(
                        model=self.llm_model,
                        previous_response_id=response.id,
                        input=next_inputs,
                        tools=[self.osworld_tool],
                        reasoning={"effort": "high"},
                    )
                except Exception as e:
                    logger.error(f"Error in tool loop continuation: {e}")
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
    
    def execute_action(self, action_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action selected by LLM.
        
        Args:
            action_info: Action information from LLM
            
        Returns:
            Observation from executing the action
        """
        logger.info(f"Executing action: {action_info['method']}")
        
        # Map shorthand methods to execute_action format
        method = action_info['method']
        params = action_info['params'].copy()
        
        # Handle shorthand action methods
        if method in ['click', 'type', 'press', 'move']:
            # Convert to execute_action format
            action_type_map = {
                'click': 'CLICK',
                'type': 'TYPING',
                'press': 'PRESS',
                'move': 'MOVE_TO'
            }
            
            params = {
                'action': {
                    'action_type': action_type_map[method],
                    'parameters': params
                }
            }
            method = 'execute_action'
        
        # Create OSWorldInteractiveAction
        action = OSWorldInteractiveAction(
            method=method,
            params=params,
            thought=action_info['goal']
        )
        
        # Execute action
        try:
            observation = self.runtime.run_action(action)
            
            # Check if it's an ErrorObservation
            if isinstance(observation, ErrorObservation):
                result = {
                    'success': False,
                    'content': observation.content[:500] if observation.content else '',
                    'exit_code': -1
                }
            else:
                # Format observation
                result = {
                    'success': getattr(observation, 'exit_code', 0) == 0,
                    'content': observation.content[:500] if observation.content else '',
                    'exit_code': getattr(observation, 'exit_code', 0)
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
    
    async def collect_trajectory(self, trajectory_id: str) -> Dict[str, Any]:
        """
        Collect a single trajectory.
        
        Args:
            trajectory_id: Unique identifier for this trajectory
            
        Returns:
            Trajectory data
        """
        logger.info(f"=" * 80)
        logger.info(f"Collecting trajectory: {trajectory_id}")
        logger.info(f"=" * 80)
        
        # Sample a persona for this trajectory
        persona = self.sample_persona()
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
        state = self.get_current_state()
        
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
            state = self.generate_action(state, goal, trajectory['steps'], trajectory_id)
           
            logger.info(f"✓ Step {step + 1} completed")
        
        trajectory['end_time'] = datetime.now().isoformat()
        trajectory['total_steps'] = len(trajectory['steps'])
        
        return trajectory
    
    def save_trajectory(self, trajectory: Dict[str, Any]):
        """
        Save trajectory data to JSON file.
        
        Args:
            trajectory: Trajectory data to save
        """
        traj_dir = self.output_dir / trajectory['trajectory_id']
        traj_file = traj_dir / 'trajectory.json'
        
        logger.info(f"Saving trajectory to {traj_file}")
        
        with open(traj_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        logger.info(f"✓ Trajectory saved: {trajectory['total_steps']} steps")
    
    async def generate_trajectories(self):
        """Generate multiple trajectories."""
        logger.info("=" * 80)
        logger.info("Starting Synthetic Data Generation")
        logger.info("=" * 80)
        
        try:
            # Initialize runtime
            await self.initialize_runtime()
            
            # Generate trajectories
            for i in range(self.max_trajectories):
                trajectory_id = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
                
                try:
                    # Collect trajectory
                    trajectory = await self.collect_trajectory(trajectory_id)
                    
                    # Save trajectory
                    self.save_trajectory(trajectory)
                    
                    logger.info(f"\n✓ Trajectory {i + 1}/{self.max_trajectories} completed\n")
                    
                    # Wait between trajectories
                    if i < self.max_trajectories - 1:
                        logger.info("Waiting 5 seconds before next trajectory...")
                        await asyncio.sleep(5)
                
                except Exception as e:
                    logger.error(f"Error collecting trajectory {trajectory_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        finally:
            # Cleanup
            await self.cleanup_runtime()
        
        logger.info("=" * 80)
        logger.info("Synthetic Data Generation Complete")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)


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
    )
    
    # Generate trajectories
    await generator.generate_trajectories()
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

