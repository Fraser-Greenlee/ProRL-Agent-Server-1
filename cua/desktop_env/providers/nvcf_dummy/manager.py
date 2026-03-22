import logging

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.nvcf_dummy.NVCFDummyVMManager")
logger.setLevel(logging.INFO)


class NVCFDummyVMManager(VMManager):
    """
    Minimal VM manager for the nvcf_dummy provider.
    This provider uses a prebuilt Docker image directly and does not need VM files.
    """

    def __init__(self, registry_path: str = ""):
        self.registry_path = registry_path

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, region=None, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, region=None, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return [""]

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, os_type: str, region: str = None, screen_size=(1920, 1080), **kwargs) -> str:
        # nvcf_dummy starts from a container image; path_to_vm is unused.
        return ""
