import logging
import os
import time

import docker
import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.nvcf_dummy.NVCFDummyProvider")
logger.setLevel(logging.INFO)

RETRY_INTERVAL = 10
WAIT_TIME = 3
DEFAULT_IMAGE = "nvcr.io/i01fc6pe8nwm/nemo:osworld-linux-eval-v2"
DEFAULT_CONTAINER_NAME = "osworld-eval"
CONTAINER_SERVER_PORT = 8000
CONTAINER_CHROMIUM_PORT = 9222


class NVCFDummyProvider(Provider):
    """
    Docker-based provider that runs a prebuilt NVCF image directly.
    Unlike the default docker provider, it does not use qcow2 VM images.
    """

    def __init__(self, region: str = None):
        super().__init__(region)
        self.client = docker.from_env()
        self.container = None
        self.container_name = None

        self.server_port = None
        self.chromium_port = None
        self.vnc_port = 8006
        self.vlc_port = 8100

    def _wait_for_vm_ready(self, timeout: int = 300):
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/api/screenshot",
                    timeout=(10, 10),
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            logger.info("Checking if nvcf_dummy container is ready...")
            time.sleep(RETRY_INTERVAL)

        raise TimeoutError("nvcf_dummy failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        # path_to_vm/headless/os_type are intentionally unused in this provider.
        del path_to_vm, headless, os_type

        image = os.environ.get("NVCF_DUMMY_IMAGE", DEFAULT_IMAGE)
        self.container_name = os.environ.get("NVCF_DUMMY_CONTAINER_NAME", DEFAULT_CONTAINER_NAME)
        self.server_port = int(os.environ.get("NVCF_DUMMY_SERVER_PORT", "8000"))
        self.chromium_port = int(os.environ.get("NVCF_DUMMY_CHROMIUM_PORT", "9222"))
        shm_size = os.environ.get("NVCF_DUMMY_SHM_SIZE", "2g")

        try:
            # Reuse existing container if present; otherwise create a new one.
            try:
                existing = self.client.containers.get(self.container_name)
                existing.reload()
                if existing.status != "running":
                    logger.info("Starting existing container '%s'.", self.container_name)
                    existing.start()
                else:
                    logger.info("Reusing running container '%s'.", self.container_name)
                self.container = existing
            except docker.errors.NotFound:
                logger.info("Creating new container '%s'.", self.container_name)
                self.container = self.client.containers.run(
                    image,
                    name=self.container_name,
                    shm_size=shm_size,
                    privileged=True,
                    ports={
                        f"{CONTAINER_SERVER_PORT}/tcp": self.server_port,
                        f"{CONTAINER_CHROMIUM_PORT}/tcp": self.chromium_port,
                    },
                    detach=True,
                )

            logger.info(
                "Started nvcf_dummy container '%s' with image '%s' (server=%s, chrome=%s)",
                self.container_name,
                image,
                self.server_port,
                self.chromium_port,
            )
            self._wait_for_vm_ready()
        except Exception as e:
            self.stop_emulator("")
            raise

    def get_ip_address(self, path_to_vm: str) -> str:
        if not self.server_port or not self.chromium_port:
            raise RuntimeError("Container not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for nvcf_dummy provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        # region/path_to_vm are kept for provider interface compatibility.
        del path_to_vm, region, args, kwargs

        try:
            target_container = self.container
            if target_container is None and self.container_name:
                try:
                    target_container = self.client.containers.get(self.container_name)
                except docker.errors.NotFound:
                    target_container = None

            if target_container is not None:
                logger.info("Stopping nvcf_dummy container '%s'...", target_container.name)
                target_container.stop()
                target_container.remove()
                time.sleep(WAIT_TIME)
        except Exception as e:
            logger.error("Error stopping nvcf_dummy container: %s", e)
        finally:
            self.container = None
            self.container_name = None
            self.server_port = None
            self.chromium_port = None
