# Configuration of Singularity

---

Welcome to the Singularity VM Management documentation.

## Overview

The Singularity provider runs OSWorld VMs using QEMU inside Singularity containers. This approach is similar to Docker but uses Singularity/Apptainer, which is commonly available on HPC clusters and doesn't require root privileges to run containers.

## Prerequisites

### 1. Install Singularity/Apptainer

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y singularity-container
```

Or install Apptainer (successor to Singularity):
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

### 2. Check KVM Support (Recommended)

KVM hardware acceleration significantly improves VM performance:

```bash
# Check if KVM is available
egrep -c '(vmx|svm)' /proc/cpuinfo  # Should return > 0

# Check KVM device
ls -la /dev/kvm

# Ensure your user has access
sudo usermod -aG kvm $USER
```

> **Note**: The provider will work without KVM but will be significantly slower.

## Setup

### 1. Build the Singularity Image

Build the runtime image from the provided definition file:

```bash
cd desktop_env/providers/singularity

# Create the singularity images directory
sudo mkdir -p /root/singularity_images

# Build the SIF image (requires sudo for some operations)
sudo singularity build /root/singularity_images/osworld_ubuntu_24_04.sif osworld_singularity.def

# Or use fakeroot if available
singularity build --fakeroot /root/singularity_images/osworld_ubuntu_24_04.sif osworld_singularity.def
```

### 2. Download VM Images

Download and extract the QCOW2 VM images to `/OS_images/`:

**Ubuntu:**
```bash
sudo mkdir -p /OS_images
cd /OS_images
sudo wget https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
sudo unzip Ubuntu.qcow2.zip
sudo rm Ubuntu.qcow2.zip
```

**Windows:**
```bash
cd /OS_images
sudo wget https://huggingface.co/datasets/xlangai/windows_osworld/resolve/main/Windows-10-x64.qcow2.zip
sudo unzip Windows-10-x64.qcow2.zip
sudo rm Windows-10-x64.qcow2.zip
```

### 3. Directory Structure

After setup, your directory should look like:

```
/root/singularity_images/
└── osworld_ubuntu_24_04.sif    # Singularity container image

/OS_images/
├── Ubuntu.qcow2                # Ubuntu VM image
└── Windows-10-x64.qcow2        # Windows VM image (optional)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SINGULARITY_VMS_DIR` | `/OS_images` | Directory containing VM images |
| `SINGULARITY_SIF_PATH` | `/root/singularity_images/osworld_ubuntu_24_04.sif` | Path to the Singularity image |
| `DISABLE_KVM` | `0` | Set to `1` to disable KVM (required for WSL2) |

## Running Experiments

Add the following arguments when initializing `DesktopEnv`:

```python
from desktop_env import DesktopEnv

env = DesktopEnv(
    provider_name="singularity",
    os_type="Ubuntu",  # or "Windows"
)
```

## Ports

The Singularity provider allocates the following ports dynamically:

| Service | Default Start | Description |
|---------|---------------|-------------|
| Server | 5000+ | OSWorld Flask server (VM internal port 5000) |
| VNC | 8006+ | VNC display (VM internal port 5900+) |
| Chromium | 9222+ | Chrome DevTools Protocol |
| VLC | 8080+ | VLC web interface |

## Troubleshooting

### "Singularity is not available"

Ensure Singularity/Apptainer is installed and in your PATH:
```bash
singularity --version
```

### "VM image not found"

Ensure the QCOW2 file exists in the correct directory:
```bash
ls -la ./singularity_vm_data/
```

### "KVM device not found"

The provider will fall back to software emulation (slower). For better performance:
```bash
# Check KVM support
sudo apt install cpu-checker
kvm-ok

# Load KVM modules
sudo modprobe kvm
sudo modprobe kvm_intel  # or kvm_amd
```

### "KVM permission denied" / "failed to initialize kvm"

This often happens in WSL2 or when KVM exists but can't be used inside Singularity:
```bash
# Option 1: Disable KVM entirely (required for WSL2)
DISABLE_KVM=1 python quickstart.py --provider_name singularity --os_type Ubuntu

# Option 2: Add user to kvm group (if KVM is actually available)
sudo usermod -aG kvm $USER
# Log out and log back in for the change to take effect
```

> **Note**: In WSL2, KVM typically cannot be used even if `/dev/kvm` exists. Use `DISABLE_KVM=1`.

### VM fails to start

Check the QEMU logs:
```bash
cat /tmp/osworld_singularity_logs/qemu_*.err
```

### Permission issues

Ensure the VM images are readable:
```bash
chmod 644 ./singularity_vm_data/*.qcow2
chmod 644 ./singularity_vm_data/*.sif
```

## Comparison with Docker Provider

| Feature | Docker | Singularity |
|---------|--------|-------------|
| Root required | Yes (daemon) | No |
| HPC compatible | Usually no | Yes |
| Image format | Docker image | SIF file |
| Networking | Docker network | Host network |
| KVM passthrough | Via devices | Via --bind |
