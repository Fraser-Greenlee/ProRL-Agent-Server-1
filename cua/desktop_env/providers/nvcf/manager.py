import logging

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.nvcf.NVCFVMManager")
logger.setLevel(logging.INFO)


class NVCFVMManager(VMManager):
    """
    Minimal VM manager for the NVCF provider.

    NVCF sessions are represented by function deployment metadata rather than
    local VM image paths, so the path value is intentionally unused.
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

    def get_vm_path(
        self,
        os_type: str,
        region: str = None,
        screen_size=(1920, 1080),
        **kwargs,
    ) -> str:
        # NVCF provider does not use local VM image paths.
        del os_type, region, screen_size, kwargs
        return ""
