import asyncio

import numpy as np
import pandas as pd

from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.nvidia.math_coder.math_utils import (
    evaluate_agent,
    initialize_agents,
    run_agent,
)
from openhands.nvidia.reward import Reward


async def run(instance):
    reward_server_ip = ['cpu-0017']
    max_iterations = 35
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-8B',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': True,
        'native_tool_calling': True,
        'temperature': 0.6,
    }
    llm_config = LLMConfig(base_url='http://127.0.0.1:8000/v1', **sampling_params)

    reward = Reward(server_ip=reward_server_ip)

    # test reward server
    test_reward = await reward.get_reward(
        instance, '<think> fake thought </think> \\boxed{025}'
    )
    logger.info(f'Test reward: {test_reward}')

    # run agent
    runtime, metadata, config = await initialize_agents(
        instance, llm_config=llm_config, max_iterations=max_iterations
    )
    run_results = await run_agent(runtime, metadata, config, instance)
    eval_results = await evaluate_agent(reward, run_results, instance)
    return eval_results


if __name__ == '__main__':
    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/deepscaler/aime.parquet'
    )
    instance = dataset.iloc[1]
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    instance = instance.to_dict()
    print(instance)

    # Initialize the agents
    results = asyncio.run(run(instance))

    logger.info(f'Run Results: {results}')
    logger.info('Agents initialized successfully!')
