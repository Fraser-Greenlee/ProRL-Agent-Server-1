"""
Test: spin up a single KVM (Singularity) VM and verify upload_file + open works.

Usage (from cua/ dir):
    python data/custom_configs/spreadsheetbench/test_upload_file_kvm.py
"""
import asyncio
import logging
import sys

sys.path.insert(0, "/lustre/fsw/portfolios/nvr/users/bcui/OSWorld")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_upload_file_kvm")

SAMPLE_XLSX = (
    "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2"
    "/cua/data/custom_configs/spreadsheetbench/all_data_912_v0.1/spreadsheet/69-17/1_69-17_input.xlsx"
)

VM_IMAGE = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/OS_images/Ubuntu.qcow2"

SAMPLE_CONFIG = {
    "id": "test-upload-file",
    "snapshot": "libreoffice_calc",
    "config": [
        {
            "type": "upload_file",
            "parameters": {
                "files": [
                    {
                        "local_path": SAMPLE_XLSX,
                        "path": "/home/user/Desktop/1_69-17_input.xlsx"
                    }
                ]
            }
        },
        {
            "type": "open",
            "parameters": {
                "path": "/home/user/Desktop/1_69-17_input.xlsx"
            }
        },
        {
            "type": "sleep",
            "parameters": {
                "seconds": 3
            }
        }
    ],
}


async def main():
    from modules_kimi.env_controller import EnvController

    logger.info("Initializing KVM runtime...")
    runtime = await EnvController.initialize_runtime(
        job_id="test-upload-file",
        vm_image_path=VM_IMAGE,
        os_type="linux",
        osworld_setup=SAMPLE_CONFIG,
        runtime_type="singularity",
    )

    try:
        logger.info("Setup succeeded! Taking screenshot...")
        screenshot = EnvController.get_screenshot(runtime)
        if screenshot:
            out_path = "/tmp/test_upload_file_screenshot.png"
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
