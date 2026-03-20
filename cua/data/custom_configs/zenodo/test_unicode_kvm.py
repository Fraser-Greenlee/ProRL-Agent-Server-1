"""
Test: verify upload_file works with unicode/special char filenames on KVM.

Usage (from cua/ dir):
    python data/custom_configs/zenodo/test_unicode_kvm.py
"""
import asyncio
import json
import logging
import sys

sys.path.insert(0, "/lustre/fsw/portfolios/nvr/users/bcui/OSWorld")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_unicode_kvm")

VM_IMAGE = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2"


def load_unicode_config():
    """Find the config at line 38 (zenodo:1233602) with apostrophe in filename."""
    config_path = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/data/custom_configs/zenodo/osworld_setup_configs.jsonl"
    with open(config_path) as f:
        for i, line in enumerate(f):
            if i == 38:
                return json.loads(line)


async def main():
    from modules_kimi.env_controller import EnvController

    sample_config = load_unicode_config()
    logger.info(f"Testing config: {sample_config['id']}")
    for c in sample_config['config']:
        if c['type'] == 'upload_file':
            logger.info(f"  Upload: {c['parameters']['files'][0]['path']}")

    logger.info("Initializing KVM runtime...")
    runtime = await EnvController.initialize_runtime(
        job_id="test-unicode",
        vm_image_path=VM_IMAGE,
        os_type="linux",
        osworld_setup=sample_config,
        runtime_type="singularity",
    )

    try:
        logger.info("Setup succeeded! Taking screenshot...")
        screenshot = EnvController.get_screenshot(runtime)
        if screenshot:
            out_path = "data/custom_configs/zenodo/test_unicode_screenshot.png"
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
