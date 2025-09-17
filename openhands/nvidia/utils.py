import asyncio
import base64
import copy
import json
import os
import queue
import subprocess
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from transformers import AutoTokenizer

try:
    from PIL import Image
except Exception:
    Image = None

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.gui_agent.gui_agent import GuiAgent
from openhands.controller.state.state import State
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.events.observation import BrowserOutputObservation
from openhands.llm.nvidia.qwen3 import (
    convert_messages_to_tokens,
    qwen3_chat_template,
)
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import JobDetails


def clear_queue(q: queue.Queue):
    """Clear all items from a queue and properly mark them as done.

    This function removes all items from the queue and calls task_done()
    for each item to maintain proper queue state for join() operations.
    """
    items_cleared = 0
    while not q.empty():
        try:
            q.get_nowait()
            try:
                q.task_done()  # Mark the task as done
                items_cleared += 1
            except ValueError as e:
                # task_done() called too many times - this shouldn't happen but handle gracefully
                logger.warning(
                    f'task_done() called too many times while clearing queue: {e}'
                )
                items_cleared += 1  # Still count the item as cleared
        except queue.Empty:
            # Queue became empty between qsize check and get_nowait
            break

    if items_cleared > 0:
        logger.debug(f'Cleared {items_cleared} items from queue')


def get_singularity_job_pids():
    all_pids = set()
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'openhands'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                all_pids.add(pid.strip())

        result = subprocess.run(
            ['pgrep', '-f', 'openhands'],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                all_pids.add(pid.strip())
    except subprocess.CalledProcessError:
        pass
    return all_pids


def kill_all_singularity_jobs(exclude_pids: set[str] | None = None):
    if exclude_pids is None:
        exclude_pids = set()
    try:
        # List all singularity-related processes
        result = subprocess.run(
            ['pgrep', '-f', 'apptainer'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            clean_pid = pid.strip()
            if clean_pid and clean_pid not in exclude_pids:
                subprocess.run(['kill', '-9', clean_pid])
                logger.info(f'Killed Singularity process with PID: {clean_pid}')
    except subprocess.CalledProcessError:
        logger.info('No Singularity processes found.')

    try:
        # List all openhands processes
        result = subprocess.run(
            ['pgrep', '-f', 'openhands'],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            clean_pid = pid.strip()
            if clean_pid and clean_pid not in exclude_pids:
                subprocess.run(['kill', '-9', clean_pid])
                logger.info(f'Killed Openhands process with PID: {clean_pid}')
    except subprocess.CalledProcessError:
        logger.info('No Openhands processes found.')


# Custom exceptions
class ServerNotRunningError(Exception):
    pass


class NoLLMServerError(Exception):
    pass


class JobTimeoutError(Exception):
    pass


class ProcessRequest(BaseModel):
    instance: dict[str, Any]
    sampling_params: dict[str, Any]
    job_id: str | None = None


class CancelRequest(BaseModel):
    job_id: str


class LLMServerRequest(BaseModel):
    address: str


async def cleanup_timed_out_job(server, job_id: str | None = None):
    """Clean up a job that has timed out or failed"""
    if job_id is None:
        return

    if not hasattr(server, '_job_details'):
        return

    if job_id in server._job_details:
        job_details = server._job_details[job_id]

        # Set timeout error flag
        job_details.timeout_error = True

        # Set the event to unblock any waiting threads
        if job_details.event is not None:
            job_details.event.set()

        # Remove from active jobs sets
        if job_id in server._active_init_jobs:
            server._active_init_jobs.remove(job_id)
        if job_id in server._active_run_jobs:
            server._active_run_jobs.remove(job_id)

        # Close runtime if it exists
        if job_details.runtime:
            try:
                job_details.runtime.close()
            except Exception:
                pass

        # Remove from job details
        if job_id in server._job_details:
            del server._job_details[job_id]


async def process_with_timeout(
    server,
    instance: pd.Series,
    sampling_params: dict[str, Any],
    timeout: float,
    thread_pool: ThreadPoolExecutor | None = None,
    job_id: str | None = None,
):
    if job_id is None:
        job_id = server.get_unique_id(instance)
    try:
        # Create a future for the process call

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            thread_pool,
            lambda: server.process(instance, sampling_params, job_id, timeout),
        )
        result = await future
        return result
    except asyncio.TimeoutError:
        # Clean up the timed-out job
        logger.error(f'Job {job_id} timed out after {timeout} seconds')
        await cleanup_timed_out_job(server, job_id)
        raise JobTimeoutError(f'Job {job_id} timed out after {timeout} seconds')
    except Exception as e:
        # Clean up on any other error
        logger.error(f'Job {job_id} failed with error: {str(e)}')
        await cleanup_timed_out_job(server, job_id)
        raise


def is_last_action_finish(state: State) -> bool:
    if state and state.history:
        last_action = next(
            (event for event in reversed(state.history) if isinstance(event, Action)),
            None,
        )
        if isinstance(last_action, AgentFinishAction):
            return True
    return False


def ngram_repetition_reward(
    sequence: np.ndarray, ngram_size: int = 64, penalty: float = -0.001
) -> np.ndarray:
    """
    Compute repetition penalty for n-grams in a sequence using NumPy.

    Args:
        sequence: 1D NumPy array of token IDs, shape (L,)
        ngram_size: int >= 1
        penalty: float (penalty for repeated n-gram occurrences except the first)

    Returns:
        reward: 1D NumPy array shape (L,) with penalty applied at the start positions of repeated n-grams.
    """
    sequence = np.asarray(sequence)
    L = sequence.shape[0]

    if ngram_size <= 0:
        raise ValueError('ngram_size must be >= 1')
    if L < ngram_size:
        return np.zeros(L, dtype=np.float32)

    # Build n-grams (rolling view)
    L - ngram_size + 1
    ngram_array = np.lib.stride_tricks.sliding_window_view(
        sequence, ngram_size
    )  # shape: (num_ngrams, ngram_size)

    # Convert each n-gram to a tuple for hashing (or use structured dtype)
    ngram_tuples = [tuple(row) for row in ngram_array]

    # Find unique ngrams and group occurrences
    from collections import defaultdict

    positions_by_ngram = defaultdict(list)
    for i, ng in enumerate(ngram_tuples):
        positions_by_ngram[ng].append(i)

    reward = np.zeros(L, dtype=np.float32)
    for positions in positions_by_ngram.values():
        if len(positions) > 1:
            # Apply penalty to all but the first occurrence
            for pos in positions[1:]:
                reward[pos] = penalty

    return reward


def process_messages_from_agent_state(
    agent: CodeActAgent,
    state: State,
    job_details: JobDetails | None = None,
) -> dict[str, Any]:
    """
    Process the messages from the agent state. We reuse logic from CodeActAgent to process state.history into litellm messages.
    We then format the messages for the LLM back to huggingface format (recognized by hf tokenizer).
    Currently Qwen3 models expect the assistant message to have a <think> tag and a </think> tag, but sometimes the agent does not end properly reasoning properly.
    We then format the message to be consistent with the expected output from Qwen3 models.

    Example of Qwen3 model that did not end properly reasoning properly:
    <think>
    <tool_call>
    {tool_call}
    </tool_call>
    ...

    We obtains the following message: {'role': 'assistant', 'content': '<think>\n', 'tool_calls': [tool_call]}

    The above breaks the hf tokenizer because it does not recognize the <think> tag.
    We thus compress all tool calls into content and append a </think> tag: {'role': 'assistant', 'content': '<think>\n{tool_call}\n{tool_call}\n</tool_call>\n</think>'}
    """
    if job_details is not None:
        assert job_details.llm_config is not None, (
            'llm_config is required in job_details.'
        )
        token_level_generation = job_details.llm_config.token_level_generation
        enable_thinking = job_details.llm_config.enable_thinking
        if token_level_generation:
            model_name = job_details.llm_config.custom_tokenizer
            # Default to Qwen3-8B if custom_tokenizer is not set.
            if model_name is None:
                model_name = 'Qwen/Qwen3-4B-Instruct-2507'
            else:
                assert 'qwen3' in model_name.lower(), (
                    'token_level_generation is only supported for Qwen3 models.'
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            chat_template = qwen3_chat_template
    else:
        logger.warning(
            'No job_details provided in process_messages_from_agent_state. Assuming token_level_generation is False.'
        )
        token_level_generation = False

    gif_paths: dict[str, str | None] = {'screenshots_gif': None, 'som_gif': None}
    should_save, trajectory_path = _should_save_screenshots()
    if (
        should_save
        and job_details is not None
        and job_details.config is not None
        and isinstance(agent, GuiAgent)
    ):
        # When should_save is True, trajectory_path is guaranteed to be a string
        assert trajectory_path is not None, (
            'trajectory_path should not be None when should_save is True'
        )
        logger.info(
            f'Saving screenshots and GIFs to trajectory path: {trajectory_path}'
        )

        # Create unique run folder with organized structure
        run_folder = _get_unique_run_folder(trajectory_path, job_details)
        som_dir = os.path.join(run_folder, 'SOM')
        screenshot_dir = os.path.join(run_folder, 'Screenshot')

        logger.debug(f'Created run folder: {run_folder}')
        logger.debug(f'SOM directory: {som_dir}')
        logger.debug(f'Screenshot directory: {screenshot_dir}')

        # Persist SoM and regular images from history to their respective directories
        _write_som_images_from_state(state, som_dir, screenshot_dir)

        # Create GIFs in the run folder (same level as SOM and Screenshot folders)
        gif_paths['screenshots_gif'] = _save_screenshots_gif(
            screenshot_dir,
            run_folder,
            gif_name='browser_run.gif',
            duration_ms=700,
            image_prefix='screenshot_',
        )
        gif_paths['som_gif'] = _save_screenshots_gif(
            som_dir,
            run_folder,
            gif_name='browser_run_som.gif',
            duration_ms=700,
            image_prefix='som_',
        )

        if gif_paths['screenshots_gif']:
            logger.info(f'Screenshots GIF saved: {gif_paths["screenshots_gif"]}')
        if gif_paths['som_gif']:
            logger.info(f'SOM GIF saved: {gif_paths["som_gif"]}')
    else:
        if not should_save:
            logger.debug('Screenshot saving disabled by environment variables')
        else:
            logger.debug('Screenshot saving skipped - no job details or config')

    # Now all agents (including GuiAgent) have _get_initial_user_message and _get_messages methods
    initial_user_message = agent._get_initial_user_message(state.history)
    raw_messages = agent._get_messages(state.history, initial_user_message)

    # For GuiAgent, keep non-assistant trailing messages to preserve context
    if (
        not isinstance(agent, GuiAgent)
        and raw_messages
        and raw_messages[-1].role != 'assistant'
    ):
        raw_messages = raw_messages[:-1]

    messages = agent.llm.format_messages_for_llm(raw_messages)

    # For GuiAgent, do not drop trailing non-assistant messages
    if not isinstance(agent, GuiAgent):
        while len(messages) > 0 and messages[-1]['role'] != 'assistant':
            messages = messages[:-1]

    from openhands.llm.llm_utils import check_tools

    tools = check_tools(agent.tools, agent.llm.config)

    new_messages = []
    for message in messages:
        new_message = {'role': message['role']}
        if isinstance(message['content'], str):
            new_message['content'] = message['content']
        elif isinstance(message['content'], list):
            if len(message['content']) == 0:
                new_message['content'] = ''  # empty string for default
            else:
                new_message['content'] = message['content'][0]['text']
        else:
            raise RuntimeError(
                f'Message content is not a string or list: {message["content"]}'
            )
        if 'tool_calls' in message:
            new_message['tool_calls'] = [
                tool_call['function'] for tool_call in message['tool_calls']
            ]

        # Update token_ids, currently this only works for Qwen3 models.
        if token_level_generation:
            input_ids = message.get('input_ids', None)
            output_ids = message.get('output_ids', None)
            logprobs = message.get('logprobs', None)
            if output_ids is not None:
                new_message['token_ids'] = output_ids
                new_message['repetition_penalty'] = ngram_repetition_reward(
                    output_ids
                ).tolist()
            else:
                new_message['token_ids'] = convert_messages_to_tokens(
                    [new_message],
                    tokenizer,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                )[0]
                new_message['repetition_penalty'] = None
            new_message['input_ids'] = input_ids
            new_message['logprobs'] = logprobs

        # Handle the case where the agent did not end properly reasoning properly
        # This largely formats the message to be consistent with the expected output from Qwen3 models.
        if (
            job_details is not None
            and job_details.agent_config['ensure_thinking_end_properly']
            and message['role'] == 'assistant'
            and '<think>' in new_message['content']
            and '</think>' not in new_message['content']
        ):
            if 'tool_calls' in new_message:
                tool_calls_message = ''
                for tool_call in new_message['tool_calls']:
                    current_tool_call = []
                    current_tool_call.append(
                        '\n<tool_call>\n{"name": "'
                        + tool_call['name']
                        + '", "arguments": '
                    )
                    if isinstance(tool_call['arguments'], str):
                        current_tool_call.append(tool_call['arguments'])
                    else:
                        current_tool_call.append(json.dumps(tool_call['arguments']))
                    current_tool_call.append('}\n</tool_call>')
                    tool_calls_message += ''.join(current_tool_call)
                new_message['content'] = (
                    f'{new_message["content"]}\n{tool_calls_message}</think>'
                )
                new_message.pop('tool_calls')
            new_messages.append(new_message)
            return {
                'messages': new_messages,
                'tools': tools,
                'end_properly': False,
            }

        new_messages.append(new_message)
    result = {
        'messages': new_messages,
        'tools': tools,
        'end_properly': not state.get_last_agent_format_error(),
    }
    # Attach GIF paths when available
    if gif_paths['screenshots_gif'] or gif_paths['som_gif']:
        result['screenshots_gif'] = gif_paths['screenshots_gif']
        result['som_gif'] = gif_paths['som_gif']
    return result


def get_messages_from_partial_result(job_details: JobDetails) -> dict[str, Any]:
    if job_details.agent is None or job_details.controller is None:
        return {'messages': [], 'tools': [], 'end_properly': True}
    controller = job_details.controller
    state = controller.get_state()
    assert state is not None, (
        'Error in get_messages_from_partial_result: state is None.'
    )
    return process_messages_from_agent_state(job_details.agent, state, job_details)


def get_instance_id(instance: dict) -> str:
    if 'instance_id' in instance:
        if instance['instance_id'] is not None:
            return instance['instance_id']
    data_source = instance.get('data_source', 'unknown')
    if 'extra_info' in instance:
        split = instance['extra_info'].get('split', 'unknown')
        index = instance['extra_info'].get('index', 'unknown')
        name = instance['extra_info'].get('name', 'unknown')
    else:
        split = 'unknown'
        index = 'unknown'
        name = 'unknown'
    return f'{data_source}_{name}_{split}_{index}'


def initialize_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in init: {str(e)}',
        'traceback': tb,
        'finish': False,
        'messages': [],
        'tools': [],
        'end_properly': False,
        'resolved': False,
        'critical_error': 'init',
    }


def run_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    git_patch = (
        job_details.run_results.get('git_patch', None)
        if job_details.run_results is not None
        else None
    )
    success = (
        job_details.run_results.get('success', False)
        if job_details.run_results is not None
        else False
    )
    finish = (
        job_details.run_results.get('finish', False)
        if job_details.run_results is not None
        else False
    )
    messages = (
        job_details.run_results.get('messages', [])
        if job_details.run_results is not None
        else []
    )
    tools = (
        job_details.run_results.get('tools', [])
        if job_details.run_results is not None
        else []
    )
    end_properly = (
        job_details.run_results.get('end_properly', True)
        if job_details.run_results is not None
        else True
    )
    if len(messages) == 0:
        partial_result = get_messages_from_partial_result(job_details)
        messages = partial_result['messages']
        tools = partial_result['tools']
        end_properly = partial_result['end_properly']
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': git_patch,
        'success': success,
        'error': f'Error in run agent: {str(e)}',
        'traceback': tb,
        'finish': finish,
        'messages': messages,
        'tools': tools,
        'end_properly': end_properly,
        'resolved': False,
        'critical_error': 'run',
    }


def eval_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    git_patch = (
        job_details.run_results.get('git_patch', None)
        if job_details.run_results is not None
        else None
    )
    success = (
        job_details.run_results.get('success', False)
        if job_details.run_results is not None
        else False
    )
    finish = (
        job_details.run_results.get('finish', False)
        if job_details.run_results is not None
        else False
    )
    messages = (
        job_details.run_results.get('messages', [])
        if job_details.run_results is not None
        else []
    )
    tools = (
        job_details.run_results.get('tools', [])
        if job_details.run_results is not None
        else []
    )
    end_properly = (
        job_details.run_results.get('end_properly', True)
        if job_details.run_results is not None
        else True
    )
    if len(messages) == 0:
        partial_result = get_messages_from_partial_result(job_details)
        messages = partial_result['messages']
        tools = partial_result['tools']
        end_properly = partial_result['end_properly']
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': git_patch,
        'success': success,
        'error': f'Error in eval: {str(e)}',
        'traceback': tb,
        'finish': finish,
        'messages': messages,
        'tools': tools,
        'end_properly': end_properly,
        'resolved': False,
        'critical_error': 'eval',
    }


def final_result(job_details: JobDetails):
    if job_details.results is None:
        if job_details.run_results is None:
            partial_result = get_messages_from_partial_result(job_details)
            messages = partial_result['messages']
            tools = partial_result['tools']
            end_properly = partial_result['end_properly']
            result = {
                'resolved': job_details.eval_results['resolved']
                if job_details.eval_results is not None
                else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
                'messages': messages,
                'tools': tools,
                'end_properly': end_properly,
            }
        else:
            result = {
                **job_details.run_results,
                'resolved': job_details.eval_results['resolved']
                if job_details.eval_results is not None
                else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
            }
    else:
        result = copy.deepcopy(job_details.results)
        if job_details.timeout_error:
            result['critical_error'] = 'timeout'

    if 'instance_id' not in result:
        instance_id = (
            job_details.instance.get('instance_id', None)
            if job_details.instance is not None
            else None
        )
        result['instance_id'] = instance_id

    if 'trajectory_id' not in result:
        trajectory_id = (
            job_details.instance.get('trajectory_id', None)
            if job_details.instance is not None
            else None
        )
        result['trajectory_id'] = trajectory_id

    # Determine if to filter the trajectory
    # Should filter if:
    # 1. Environment error (critical_error is not None)
    # 2. Length exceeded (normally AgentFormatError)
    def determine_filter(result: dict) -> bool:
        if 'critical_error' in result and result['critical_error'] is not None:
            return True
        if 'success' in result and result['success']:
            return False
        # Waive AgentToolCallError they should be trained
        if 'error' in result:
            for normal_error in ['AgentToolCallError']:
                if normal_error in result['error']:
                    return False
        return True

    result['filter'] = determine_filter(result)
    return result


def _save_screenshots_gif(
    images_dir: str,
    output_dir: str,
    gif_name: str = 'browser_run.gif',
    duration_ms: int = 700,
    image_prefix: str = 'screenshot_',
) -> str | None:
    """Create a GIF from images in the specified directory.

    Args:
        images_dir: Directory containing the images to create GIF from
        output_dir: Directory to save the GIF file
        gif_name: Name of the output GIF file
        duration_ms: Duration of each frame in milliseconds
        image_prefix: Prefix to filter image files by

    Returns:
        Path to the created GIF file, or None if creation failed
    """
    try:
        if not os.path.isdir(images_dir):
            return None

        all_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        files = [f for f in all_files if f.startswith(image_prefix)]
        if not files:
            return None
        files.sort()

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, gif_name)

        if Image is None:
            return None

        frames = []
        for fname in files:
            p = os.path.join(images_dir, fname)
            try:
                img = Image.open(p).convert('RGB')
                frames.append(img)
            except Exception:
                continue
        if not frames:
            return None

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            quality=85,
            format='GIF',
        )
        return output_path
    except Exception:
        return None


def _write_som_images_from_state(
    state: State, som_dir: str, screenshot_dir: str
) -> None:
    """Write SOM and regular screenshots to their respective directories.

    Args:
        state: The agent state containing browser observations
        som_dir: Directory to save SOM (Set of Marks) images
        screenshot_dir: Directory to save regular screenshots
    """
    try:
        os.makedirs(som_dir, exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
    except Exception as e:
        logger.warning(
            f'Failed to create directories {som_dir} and {screenshot_dir}: {e}'
        )
        return

    som_idx = 0
    screenshot_idx = 0

    for event in state.history:
        if isinstance(event, BrowserOutputObservation):
            # Handle SOM images
            som = getattr(event, 'set_of_marks', None)
            if som:
                try:
                    screenshot_path = getattr(event, 'screenshot_path', None)
                    if screenshot_path is not None:
                        stem = Path(screenshot_path).stem
                        # Prefer timestamp from screenshot filename if present
                        ts = stem.replace('screenshot_', '')
                    else:
                        ts = time.strftime('%Y%m%d_%H%M%S_%f')
                    fname = f'som_{ts}_{som_idx}.png'
                    som_idx += 1
                    som_path = os.path.join(som_dir, fname)
                    data = som.replace('data:image/png;base64,', '')
                    img_bytes = base64.b64decode(data)
                    with open(som_path, 'wb') as f:
                        f.write(img_bytes)
                except Exception:
                    continue

            # Handle regular screenshots
            screenshot_b64 = getattr(event, 'screenshot', '')
            if screenshot_b64:
                try:
                    screenshot_path2 = getattr(event, 'screenshot_path', None)
                    if screenshot_path2 is not None:
                        stem2 = Path(screenshot_path2).stem
                        ts2 = stem2.replace('screenshot_', '')
                    else:
                        ts2 = time.strftime('%Y%m%d_%H%M%S_%f')
                    fname2 = f'screenshot_{ts2}_{screenshot_idx}.png'
                    screenshot_idx += 1
                    screenshot_path = os.path.join(screenshot_dir, fname2)
                    # Support both data URI and raw base64
                    if screenshot_b64.startswith('data:image'):
                        screenshot_b64 = screenshot_b64.split(',', 1)[1]
                    img_bytes2 = base64.b64decode(screenshot_b64)
                    with open(screenshot_path, 'wb') as f:
                        f.write(img_bytes2)
                except Exception:
                    continue

    logger.debug(f'Saved {som_idx} SOM images to {som_dir}')
    logger.debug(f'Saved {screenshot_idx} regular screenshots to {screenshot_dir}')


def _get_unique_run_folder(
    base_trajectory_path: str, job_details: JobDetails | None = None
) -> str:
    """Create a unique folder for this run with organized subfolders.

    Returns the path to the unique run folder containing:
    - SOM/ (for SOM screenshots)
    - Screenshot/ (for regular screenshots)
    - GIFs will be saved at the same level as these folders
    """
    # Generate unique identifier for this run
    if job_details and job_details.instance:
        instance_id = job_details.instance.get('instance_id', None)
        if instance_id:
            run_id = f'{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        else:
            run_id = f'run_{int(time.time())}_{str(uuid.uuid4())[:8]}'
    else:
        run_id = f'run_{int(time.time())}_{str(uuid.uuid4())[:8]}'

    run_folder = os.path.join(base_trajectory_path, run_id)

    # Create the folder structure
    try:
        os.makedirs(os.path.join(run_folder, 'SOM'), exist_ok=True)
        os.makedirs(os.path.join(run_folder, 'Screenshot'), exist_ok=True)
    except Exception as e:
        logger.warning(f'Failed to create run folder structure: {e}')
        return run_folder

    return run_folder


def _should_save_screenshots() -> tuple[bool, str | None]:
    """Check if screenshots should be saved based on environment variables.

    Returns:
        tuple: (should_save, trajectory_path)
    """
    save_screenshots = os.environ.get(
        'OH_SAVE_SCREENSHOTS_IN_TRAJECTORY', 'false'
    ).lower() in ('1', 'true', 'yes')
    trajectory_path = os.environ.get('OH_SAVE_TRAJECTORY_PATH')

    if not save_screenshots:
        return False, None

    if not trajectory_path or not os.path.exists(trajectory_path):
        logger.warning(
            f"OH_SAVE_TRAJECTORY_PATH not set or path doesn't exist: {trajectory_path}"
        )
        return False, None

    return True, trajectory_path
