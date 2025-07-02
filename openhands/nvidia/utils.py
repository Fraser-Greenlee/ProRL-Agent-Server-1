import asyncio
import json
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from pydantic import BaseModel

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.nvidia.logger import nvidia_logger as logger


def clear_queue(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


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


class LLMServerRequest(BaseModel):
    address: str


async def cleanup_timed_out_job(server, job_id: str | None = None):
    """Clean up a job that has timed out or failed"""
    if job_id is None:
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

        # Wait for the future with timeout
        # We use 2x the timeout to account time in queue
        result = await asyncio.wait_for(future, timeout=timeout * 2)
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


def process_messages_from_agent_state(
    agent: CodeActAgent, state: State
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
    initial_user_message = agent._get_initial_user_message(state.history)
    raw_messages = agent._get_messages(state.history, initial_user_message)
    messages = agent.llm.format_messages_for_llm(raw_messages)
    if messages[-1]['role'] != 'assistant':
        messages = messages[:-1]

    from openhands.llm.llm_utils import check_tools

    tools = check_tools(agent.tools, agent.llm.config)

    new_messages = []
    for message in messages:
        new_message = {'role': message['role']}
        if isinstance(message['content'], str):
            new_message['content'] = message['content']
        else:
            new_message['content'] = message['content'][0]['text']
        if 'tool_calls' in message:
            new_message['tool_calls'] = [
                tool_call['function'] for tool_call in message['tool_calls']
            ]

        # Handle the case where the agent did not end properly reasoning properly
        # This largely formats the message to be consistent with the expected output from Qwen3 models.
        if (
            message['role'] == 'assistant'
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
    return {
        'messages': new_messages,
        'tools': tools,
        'end_properly': True,
    }
