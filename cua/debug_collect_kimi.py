"""
Entry point for Kimi-K2.5 single-trajectory debugging.

Usage:
    python debug_collect_kimi.py --model_node pool0-03161
"""
import argparse
import asyncio
import logging

from openhands.core.logger import openhands_logger
from modules_kimi.data_collector import DataCollector


openhands_logger.setLevel(logging.WARNING)
logger = openhands_logger.getChild('collect_kimi')
logger.setLevel(logging.DEBUG)

for handler in logging.root.handlers:
    handler.setLevel(logging.DEBUG)

if openhands_logger.handlers:
    for handler in openhands_logger.handlers:
        handler.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="Kimi-K2.5 trajectory collection (debug)")

    # Kimi vLLM server
    parser.add_argument("--model_node", type=str, required=True,
                        help="Hostname of the Kimi vLLM server head node")
    parser.add_argument(
        "--kimi_model_name", type=str,
        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Kimi-K2.5",
    )

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

    return parser.parse_args()


async def main(args):
    data_collector = DataCollector(args)
    await data_collector.single_trajectory_job(trajectory_idx=123456)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
