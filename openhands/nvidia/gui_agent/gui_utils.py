from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, cast

from evaluation.utils.shared import (  # type: ignore
    EvalException,
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)

from openhands.core.config import AgentConfig, OpenHandsConfig
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as openhands_logger
from openhands.core.main import create_runtime
from openhands.core.setup import create_agent, create_controller
from openhands.controller.state.state import State
from openhands.events.action import MessageAction
from openhands.nvidia.controller import run_controller_with_controller
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import JobDetails, _DEFAULT_AGENT_CONFIG
from openhands.nvidia.utils import (
    get_instance_id,
    is_last_action_finish,
    process_messages_from_agent_state,
)
from openhands.runtime.base import Runtime


def get_config(
    instance: dict,
    metadata: EvalMetadata,
    agent_config: dict = _DEFAULT_AGENT_CONFIG,
) -> OpenHandsConfig:
    """Build OpenHandsConfig for GuiAgent GUI tasks.

    Key differences vs math/code configs:
    - default_agent is GuiAgent
    - browsing is enabled with SoM visual browsing
    - jupyter/cmd/editor disabled
    - use general browsing (no browsergym_eval_env preset)
    """
    sandbox_config = get_default_sandbox_config_for_eval()
    # Ensure general-purpose browsing (no fixed benchmark env)
    sandbox_config.browsergym_eval_env = None
    sandbox_config.runtime_container_image = "/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/singularity_images_oh/oh_v0.40.0_8me96m20iqt6tw9p_t5sffwjb6stny0ze.sif"
    # Browsing often benefits from host networking for external access
    # Keep the default as-is; caller environment may override via env vars

    config = OpenHandsConfig(
        default_agent='GuiAgent',
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.save_trajectory_path = os.environ.get("OH_SAVE_TRAJECTORY_PATH")  # 例如 /tmp/oh_trajs 或 /tmp/oh_trajs/run.json
    config.save_screenshots_in_trajectory = os.environ.get(
        "OH_SAVE_SCREENSHOTS_IN_TRAJECTORY", "false"
    ).lower() in ("1", "true", "yes")

    # LLM logging augmentation
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, get_instance_id(instance)
        )
    )

    agent_cfg = AgentConfig(
        enable_jupyter=False,
        enable_editor=False,
        enable_cmd=False,
        enable_browsing=True,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_think=False,
        enable_history_truncation=False,
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'],
        action_timeout=30.0,
        strict_loop_detector=agent_config['strict_loop_detector'],
        enable_som_visual_browsing=True,
    )
    config.set_agent_config(agent_cfg)

    # Optional: mount OpenHands source into runtime if provided
    mount_dir_env = os.environ.get('OVERWRITE_OPENHANDS_DIR', '').strip()
    if mount_dir_env:
        try:
            mount_path = Path(mount_dir_env).expanduser().resolve()
            if mount_path.exists() and mount_path.is_dir():
                config.sandbox.volumes = f"{mount_path}:/openhands/code:ro"
            else:
                logger.warning(
                    f"OVERWRITE_OPENHANDS_DIR is not a valid directory: {mount_dir_env}. Skipping mount."
                )
        except Exception as e:
            logger.error(
                f"Failed to set mount from OVERWRITE_OPENHANDS_DIR='{mount_dir_env}': {e}. Skipping mount."
            )

    return config


def get_instruction(instance: dict, metadata: EvalMetadata) -> MessageAction:
    """Construct initial instruction for the GUI/VLM browsing task."""
    task = instance.get('task') or instance.get('goal') or 'Open Google and search for NVIDIA stock price.'
    start_url = instance.get('start_url', None)

    instruction = """
You are a GUI web-browsing assistant with vision. Use the browsing tools to complete the task. Prefer visible, clickable elements identified by bid.
- Think step by step before acting.
- If a page is still loading, briefly wait using noop(ms).
- Interact only with visible elements.

Task:
{task}
{start_url_hint}
""".strip().format(
        task=task,
        start_url_hint=(f"Start from: {start_url}" if start_url else ''),
    )

    return MessageAction(content=instruction)


async def initialize_agents(
    instance: dict,
    llm_config: LLMConfig | None = None,
    sid: str | None = None,
    eval_output_dir: str = '/root',
    git_commit: str = 'unknown',
    dataset: str = 'gui',
    data_split: str = 'mock',
    agent_config: dict = dict(_DEFAULT_AGENT_CONFIG),
) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    metadata = EvalMetadata(
        agent_class='VisualBrowsingAgent',
        llm_config=llm_config,
        agent_config=None,
        max_iterations=agent_config['max_iterations'],
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details=None,
        condenser_config=NoOpCondenserConfig(),
    )

    config = get_config(instance, metadata, agent_config)
    runtime = create_runtime(config, sid=sid)
    await runtime.connect()
    logger.debug(f"Runtime connected {runtime.sid}")

    try:
        initialize_runtime(runtime, instance, metadata)
    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    return runtime, metadata, config


def initialize_runtime(runtime: Runtime, instance: dict, metadata: EvalMetadata):
    """Optional runtime initialization for browsing tasks.

    For general web tasks, no special container setup is required beyond connecting.
    """
    openhands_logger.info(f'{"-" * 50} BEGIN GUI Runtime Initialization {"-" * 50}')
    # Intentionally minimal – VisualBrowsingAgent will start with noop to obtain first observation
    openhands_logger.info(f'{"-" * 50} END GUI Runtime Initialization {"-" * 50}')


async def run_agent(
    job_details: JobDetails,
    sid: str | None = None,
) -> dict[str, object]:
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    # Narrow Optional types before use
    if instance is None:
        raise EvalException('Instance is None')
    if metadata is None:
        raise EvalException('Metadata is None')
    if config is None:
        raise EvalException('Config is None')
    if runtime is None:
        raise EvalException('Runtime is None')

    message_action = get_instruction(instance, metadata)

    agent = None
    state: State | None = None

    try:
        agent = create_agent(config)
        # JobDetails.agent is typed as CodeActAgent | None but GUI uses a different agent
        job_details.agent = cast(Any, agent)
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller

        state = await run_controller_with_controller(
            config=config,
            initial_user_action=message_action,
            sid=sid,
            runtime=runtime,
            agent=agent,
            fake_user_response_fn=None,
            controller=controller,
            initial_state=initial_state,
        )
        if state is None:
            raise EvalException('Final state is None')

    except Exception as e:
        logger.error(f"Error running GUI agent: {e}")

    # Extract messages from agent state
    try:
        run_results = process_messages_from_agent_state(agent, state, job_details)  # type: ignore
    except Exception as e:
        logger.error(f"Error while running GUI agent: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    # For GUI tasks, do not treat headless max-iteration as an error. Preserve messages.
    success = True
    error_msg = None
    finish_flag = is_last_action_finish(state) if state is not None else False
    if state is not None and state.last_error:
        # Detect the specific headless max-iteration pattern
        if 'Agent reached maximum iteration in headless mode' in state.last_error:
            success = True
            error_msg = None
        else:
            success = False
            error_msg = state.last_error

    return {
        'success': success,
        'error': error_msg,
        'finish': finish_flag,
        **run_results,
    }


async def evaluate_agent(run_results: dict, instance: dict) -> dict[str, Any]:
    """Trivial evaluator for GUI tasks.

    Marks resolved True if the agent finished without error; otherwise False.
    """
    success = bool(run_results.get('success', False))
    finish = bool(run_results.get('finish', False))
    return {'resolved': success and finish}
