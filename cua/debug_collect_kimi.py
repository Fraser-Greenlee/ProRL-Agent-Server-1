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
    parser.add_argument("--generation_mode", type=str, default="vanilla",
                        choices=["vanilla", "spreadsheetbench", "zenodo"],
                        help="Data generation mode")
    parser.add_argument("--max_steps_per_trajectory", type=int, default=100)
    parser.add_argument("--trajectory_save_dir", type=str,
                        default="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/trajectories/kimi_debug")

    args = parser.parse_args()

    PROJECT_DIR = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua"

    if args.generation_mode == "vanilla":
        args.persona_dataset_path = "/lustre/fsw/portfolios/nvr/users/yidong/data/nemotron_data/data/"
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = "/lustre/fsw/portfolios/nvr/users/mingjiel/data/osworld/osworld_test_nogdrive.json"
    elif args.generation_mode == "spreadsheetbench":
        args.persona_dataset_path = None
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = f"{PROJECT_DIR}/data/custom_configs/spreadsheetbench/osworld_setup_configs.jsonl"
    elif args.generation_mode == "zenodo":
        args.persona_dataset_path = None
        args.example_instructions_path = f"{PROJECT_DIR}/data/agentnet/processed/instructions.txt"
        args.osworld_setup_path = f"{PROJECT_DIR}/data/custom_configs/zenodo/osworld_setup_configs.jsonl"
    else:
        raise NotImplementedError

    return args


async def main(args):
    data_collector = DataCollector(args)
    await data_collector.single_trajectory_job(trajectory_idx=1234)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
