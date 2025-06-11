import time
import pandas as pd
import numpy as np
import os
import asyncio

from evaluation.benchmarks.swe_bench.run_infer import (
    get_config, 
    initialize_runtime, 
    get_instruction, 
    complete_runtime, 
    is_fatal_evaluation_error,
    codeact_user_response, 
    EvalException
)
from evaluation.utils.shared import (
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)
from evaluation.benchmarks.swe_bench.resource.mapping import get_instance_resource_factor

from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
)
from openhands.core.main import create_runtime, run_controller
from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    base_container_image = get_instance_docker_image(
        instance['instance_id']
    )
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.runtime_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )

    # https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/agent_config.py

    # Think Tool
    # https://github.com/yidong72/OpenHands_internal/blob/5fe7578f45a847f39e6a22947b504f2167e98117/openhands/agenthub/codeact_agent/tools/think.py#L14
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=False, # not too sure what this does. 
    )
    config.set_agent_config(agent_config)
    return config

async def initialize_agents(
        instance:pd.Series, 
        llm_config:LLMConfig,
        eval_output_dir:str = "/root",
        git_commit:str = "9f93e8a1532d6e1da4ea702f3dbd31d0f6b2fb3a", 
        dataset:str = "swebench", 
        data_split:str = "train", 
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    """
    llm_config = LLMConfig(
        model="openai/Qwen/Qwen3-8B",
        base_url="http://127.0.0.1:8000/v1",
        api_key="mykey",
        modify_params=False,
        log_completions=True,
        native_tool_calling=True,
        temperature=0.6,
    )
    """
    metadata = EvalMetadata(
        agent_class="CodeActAgent",
        llm_config=llm_config,
        agent_config=None,
        max_iterations=50,
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details={'mode': 'swe'},
        condenser_config=NoOpCondenserConfig(),
    )

    
    config = get_config(instance, metadata)

    metadata.details['runtime_failure_count'] = 0
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config)

    await runtime.connect()

    try:
        initialize_runtime(runtime, instance, metadata)

    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    return runtime, metadata, config, instance

async def run_agent(
        runtime:Runtime, 
        metadata:EvalMetadata, 
        config:OpenHandsConfig, 
        instance:pd.Series
    ) -> str:
    message_action = get_instruction(instance, metadata)
    try:
        state: State | None = await run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
            )

         # if fatal error, throw EvalError to trigger re-run
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)
        
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        #raise e
        runtime.close()
        return ""
    runtime.close()
    return git_patch

async def run(instance):
    #agent = initialize_agents(instance)
    #await initialize_runtime_for_agent(agent, instance)
    #return await run_agent(agent, instance)
    runtime, metadata, config = await initialize_agents(instance)
    results  = await run_agent(runtime, metadata, config, instance)
    return results

if __name__ == "__main__":

    dataset = pd.read_parquet("/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet")
    
    instance = dataset.iloc[0]['instance']
    # convert instance to serializable pandas series
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(run(instance))
    print("BEGIN RESULTS.")
    print(result)
    print("END RESULTS.")