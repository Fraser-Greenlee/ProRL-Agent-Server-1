"""
Test: spin up a single NVCF DesktopEnv instance and verify upload_file + open works.

Usage (from cua/ dir):
    python data/custom_configs/spreadsheetbench/test_upload_file_nvcf.py
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_upload_file_nvcf")

# OSWorld path
_osworld_path = "/lustre/fsw/portfolios/nvr/users/bcui/OSWorld"
if _osworld_path not in sys.path:
    sys.path.insert(0, _osworld_path)

SAMPLE_XLSX = (
    "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2"
    "/cua/data/custom_configs/spreadsheetbench/all_data_912_v0.1/spreadsheet/69-17/1_69-17_input.xlsx"
)

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


def main():
    from desktop_env.desktop_env import DesktopEnv

    logger.info("Creating NVCF DesktopEnv...")
    env = DesktopEnv(
        provider_name="nvcf",
        path_to_vm="",
        action_space="pyautogui",
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
    )

    try:
        logger.info("Resetting with upload_file config...")
        env.reset(task_config=SAMPLE_CONFIG)
        logger.info("Reset succeeded!")

        logger.info("Taking screenshot to verify...")
        screenshot = env.controller.get_screenshot()
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
        logger.info("Closing DesktopEnv...")
        env.close()


if __name__ == "__main__":
    main()
