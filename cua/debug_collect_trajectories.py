import argparse
import asyncio
import logging

from openhands.core.logger import openhands_logger
from modules.module_data_collector import DataCollector


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
    parser.add_argument("--planner_node", type=str, required=True)
    parser.add_argument("--actor_node", type=str, required=True)
    # DataCollector
    parser.add_argument(
        "--vm_image_path", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2",
    )
    parser.add_argument(
        "--persona_dataset_path", type=str,
        default="/lustre/fsw/portfolios/nvr/users/yidong/data/nemotron_data/data/",
    )
    parser.add_argument(
        "--example_instructions_path", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/data/agentnet/processed/instructions.txt",
    )
    parser.add_argument(
        "--osworld_setup_path", type=str,
        default="/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json",
    )
    parser.add_argument("--max_steps_per_trajectory", type=int, default=150)
    parser.add_argument("--max_steps_per_goal", type=int, default=10)

    # Planner
    parser.add_argument(
        "--planner_model_name", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"
    )
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=5120 * 28 * 28)
    parser.add_argument("--max_retry_for_goal_generation", type=int, default=1)

    # Actor
    parser.add_argument("--actor_model_name", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--max_retry_for_action_generation", type=int, default=3)

    args = parser.parse_args()

    return args


async def main(args):
    data_collector = DataCollector(args)
    await data_collector.single_trajectory_job(trajectory_idx=123456)


if __name__ == "__main__":
    # for debugging purposes, re-create "./trajectories" folder
    import os, shutil
    trajectory_dir = "./trajectories"
    shutil.rmtree(trajectory_dir)
    os.makedirs(trajectory_dir)

    args = parse_args()
    asyncio.run(main(args))




