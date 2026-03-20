"""
Test: spin up a single KVM (Singularity) VM with a Zenodo pptx config and verify it works.

Usage (from cua/ dir):
    python data/custom_configs/zenodo/test_upload_file_kvm.py
"""
import asyncio
import json
import logging
import sys

sys.path.insert(0, "/lustre/fsw/portfolios/nvr/users/bcui/OSWorld")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_zenodo_kvm")

VM_IMAGE = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2"


def load_sample_config():
    config_path = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/data/custom_configs/zenodo/osworld_setup_configs.jsonl"
    with open(config_path) as f:
        return json.loads(f.readline())


async def main():
    from modules_kimi.env_controller import EnvController

    sample_config = load_sample_config()
    logger.info(f"Testing config: {sample_config['id']}")
    logger.info(f"Config steps: {[c['type'] for c in sample_config['config']]}")

    logger.info("Initializing KVM runtime...")
    runtime = await EnvController.initialize_runtime(
        job_id="test-zenodo",
        vm_image_path=VM_IMAGE,
        os_type="linux",
        osworld_setup=sample_config,
        runtime_type="singularity",
    )

    try:
        logger.info("Setup succeeded! Taking screenshot...")
        screenshot = EnvController.get_screenshot(runtime)
        if screenshot:
            out_path = "data/custom_configs/zenodo/test_zenodo_screenshot.png"
            with open(out_path, "wb") as f:
                f.write(screenshot)
            logger.info(f"Screenshot saved to {out_path}")
        else:
            logger.warning("No screenshot returned")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Closing runtime...")
        runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
