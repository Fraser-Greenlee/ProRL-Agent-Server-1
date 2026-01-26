import argparse
import asyncio
import logging

from openhands.core.logger import openhands_logger
from modules.debug_data_collector import DataCollector


# Create a child logger
openhands_logger.setLevel(logging.WARNING)  # todo logging.WARNING
logger = openhands_logger.getChild('collect_trajectories')
logger.setLevel(logging.DEBUG)

for handler in logging.root.handlers:
    handler.setLevel(logging.DEBUG)

# If openhands_logger has its own specific handlers, update them too:
if openhands_logger.handlers:
    for handler in openhands_logger.handlers:
        handler.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()

    # Set Explorer / Parser nodes
    parser.add_argument("--explorer_node", type=str)
    parser.add_argument("--parser_node", type=str, required=True)
    parser.add_argument(
        "--explorer_model_name", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking")

    # DataCollector
    parser.add_argument(
        "--vm_image_path", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2",
    )
    parser.add_argument(
        "--persona_dataset_path", type=str,
        default="/lustre/fsw/portfolios/nvr/users/yidong/data/nemotron_data/data/"
    )
    parser.add_argument(
        "--osworld_setup_path", type=str,
        default="/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json"
    )
    parser.add_argument("--max_steps_per_trajectory", type=int, default=100)
    parser.add_argument("--max_steps_per_goal", type=int, default=5)

    # OpenAIWrapper
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=-5120 * 28 * 28)
    parser.add_argument("--max_retry_for_goal_generation", type=int, default=1)
    parser.add_argument("--max_retry_for_action_generation", type=int, default=3)


    args = parser.parse_args()

    return args


async def main(args):
    data_collector = DataCollector(args)
    await data_collector.submit_trajectory_job(123456)


if __name__ == "__main__":
    # for debugging purposes, re-create "./trajectories" folder
    import os, shutil
    trajectory_dir = "./trajectories"
    shutil.rmtree(trajectory_dir)
    os.makedirs(trajectory_dir)

    args = parse_args()
    asyncio.run(main(args))


# How to run (interactive session)
# 1. Run parser server (+ vLLM server)
#   cd ~/lustre/cua/prorl-agent-server/cua/scripts; sbatch run_models.sbatch
# 2. Check the parser server / vLLM server log to fetch node names
# 3. Boot kvm-writable CPU node
#   cd ~/lustre/cua/prorl-agent-server/cua/scripts; bash debug_interactive.sh
# 4. Run data collection script with
#   python debug_collect_trajectories.py --parser_node=pool0-02237 --explorer_node=pool0-03094



