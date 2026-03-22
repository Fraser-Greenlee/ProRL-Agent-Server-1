import os
import logging

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.singularity.SingularityVMManager")
logger.setLevel(logging.INFO)

VMS_DIR = os.environ.get("SINGULARITY_VMS_DIR", "/root/osworld/")

UBUNTU_VM_NAME = "Ubuntu.qcow2"
WINDOWS_VM_NAME = "Windows-10-x64.qcow2"


class SingularityVMManager(VMManager):
    """Minimal VM Manager for Singularity provider.
    
    This manager assumes VM images are already downloaded and available.
    Set SINGULARITY_VMS_DIR environment variable to specify the VM directory.
    
    Expected directory structure:
        /OS_images/                    (default, or $SINGULARITY_VMS_DIR)
            Ubuntu.qcow2
            Windows-10-x64.qcow2
        
        /root/singularity_images/      (default, or $SINGULARITY_SIF_PATH)
            osworld_ubuntu_24_04.sif
    """
    
    def __init__(self, registry_path: str = ""):
        self.vms_dir = VMS_DIR
        os.makedirs(self.vms_dir, exist_ok=True)

    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, region=None, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, region=None, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        """List available VM images."""
        vms = []
        for vm_name in [UBUNTU_VM_NAME, WINDOWS_VM_NAME]:
            vm_path = os.path.join(self.vms_dir, vm_name)
            if os.path.exists(vm_path):
                vms.append(vm_path)
        return vms

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, os_type: str, region: str = None, screen_size=(1920, 1080), **kwargs) -> str:
        """Get path to VM image based on OS type.
        
        Args:
            os_type: "Ubuntu" or "Windows"
            region: Ignored for Singularity provider
            screen_size: Ignored for Singularity provider
            
        Returns:
            Path to the QCOW2 VM image
            
        Raises:
            FileNotFoundError: If the VM image is not found
        """
        if os_type.lower() == "ubuntu":
            vm_name = UBUNTU_VM_NAME
        elif os_type.lower() == "windows":
            vm_name = WINDOWS_VM_NAME
        else:
            raise ValueError(f"Unsupported OS type: {os_type}. Use 'Ubuntu' or 'Windows'.")
        
        vm_path = os.path.join(self.vms_dir, vm_name)
        
        if not os.path.exists(vm_path):
            raise FileNotFoundError(
                f"VM image not found: {vm_path}\n"
                f"Please download the VM image manually.\n"
                f"Ubuntu: https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip\n"
                f"Windows: https://huggingface.co/datasets/xlangai/windows_osworld/resolve/main/Windows-10-x64.qcow2.zip\n"
                f"Extract to: {self.vms_dir}"
            )
        
        return vm_path
