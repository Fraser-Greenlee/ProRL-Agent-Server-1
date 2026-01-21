import argparse
import asyncio
import logging

from openhands.core.logger import openhands_logger
from modules.debug_data_collector import DataCollector


# Create a child logger
openhands_logger.setLevel(logging.WARNING)
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

    args = parser.parse_args()

    return args


async def main(args):
    data_collector = DataCollector(args)
    await data_collector.submit_trajectory_job(123456)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))


