# OSWorld Linux Docker Environment

This Docker image replicates the OSWorld Linux environment (Ubuntu 22.04 with GNOME Shell desktop) for benchmark tasks. It includes all required applications and configurations to match the original OSWorld QEMU VM exactly.

## Screenshot

![Ubuntu GNOME Shell Desktop](screenshot_gnome_desktop.png)

*Ubuntu 22.04 with GNOME Shell desktop environment - matching the original OSWorld QEMU VM*

## Features

- **Base OS**: Ubuntu 22.04 LTS with GNOME Shell 42.9 (same as original OSWorld VM)
- **Display**: 1920x1080 resolution via Xvfb (headless)
- **VNC Access**: x11vnc + noVNC for remote desktop access
- **OSWorld Server**: Flask-based API server for automation
- **Caddy Reverse Proxy**: Unified access to all services on port 8000
- **No Privileged Mode**: Runs without `--privileged` or `SYS_ADMIN` capability (NVCF compatible)

### Available Images

| Image | Dockerfile | Description |
|-------|------------|-------------|
| `osworld-linux` | `Dockerfile` | GNOME Shell (English UI) - matches original OSWorld VM |
| `osworld-linux-zh` | `Dockerfile.chinese-gnome` | Simplified Chinese UI (简体中文) |

### Installed Applications

| Application | Version | Purpose |
|-------------|---------|---------|
| Google Chrome | Latest stable | Web browser tasks (46 benchmark tasks) |
| LibreOffice | 7.x | Office suite tasks |
| GIMP | 2.10.x | Image editing tasks (26 benchmark tasks) |
| VLC Media Player | 3.0.x | Media playback tasks (17 benchmark tasks) |
| Visual Studio Code | Latest | Code editing tasks (23 benchmark tasks) |
| Thunderbird | Latest | Email client tasks (15 benchmark tasks) |

### LibreOffice Components
- LibreOffice Writer - Document processing (23 benchmark tasks)
- LibreOffice Calc - Spreadsheet operations (47 benchmark tasks)
- LibreOffice Impress - Presentation editing (47 benchmark tasks)

## Quick Start

### Build the Image

```bash
cd osworld-docker

# Build the GNOME Shell image (recommended - matches original OSWorld VM)
docker build -t osworld-linux .

# Chinese version
docker build -t osworld-linux-zh -f Dockerfile.chinese-gnome .
```

### Run with Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Run with Docker

```bash
docker run -d --name osworld \
  --shm-size=2g \
  -p 8000:8000 \
  -p 9222:9222 \
  osworld-linux
```

For the Chinese version:
```bash
docker run -d --name osworld-zh \
  --shm-size=2g \
  -p 8000:8000 \
  -p 9222:9222 \
  osworld-linux-zh
```

**Note**: No `--privileged` flag or special capabilities are required. The image uses a mock logind D-Bus service to run GNOME Shell without systemd, making it compatible with NVCF and other restricted container environments.

## Accessing the Environment

All services are accessible through the Caddy reverse proxy on port 8000:

### Web-based VNC (noVNC)
Open your browser and navigate to:
```
http://localhost:8000/vnc.html
```
Or simply:
```
http://localhost:8000
```

### OSWorld API Server
The REST API server is available at:
```
http://localhost:8000/api/
```

Endpoints:
- `GET /api/screenshot` - Get current desktop screenshot (PNG)
- `POST /api/execute` - Execute a command
- `POST /api/setup` - Setup configuration
- `GET /api/info` - Get system information
- `GET /api/accessibility_tree` - Get accessibility tree

### Chrome DevTools Protocol
Chrome is configured with remote debugging enabled:
```
http://localhost:8000/chrome/json
```
Or directly on port 9222:
```
http://localhost:9222
```

### VLC HTTP Interface
```
http://localhost:8000/vlc/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                         │
│                  (no privileged mode)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Caddy :8000                       │    │
│  │  ┌─────────┬────────┬────────┬─────────┬─────────┐  │    │
│  │  │ /vnc/*  │ /api/* │/chrome/│  /vlc/* │    /    │  │    │
│  │  └────┬────┴───┬────┴───┬────┴────┬────┴────┬────┘  │    │
│  └───────┼────────┼────────┼─────────┼─────────┼───────┘    │
│          │        │        │         │         │            │
│          ▼        ▼        ▼         ▼         ▼            │
│      ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌────────┐       │
│      │noVNC │ │Server│ │Chrome│ │  VLC  │ │Redirect│       │
│      │:5910 │ │:5000 │ │:9222 │ │ :8100 │ │→noVNC  │       │
│      └──────┘ └──────┘ └──────┘ └───────┘ └────────┘       │
│          │                                                  │
│          ▼                                                  │
│      ┌──────────────────────────────────────────────┐      │
│      │              GNOME Shell Desktop             │      │
│      │                   (mutter)                   │      │
│      │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐   │      │
│      │  │Chrome│ │ VLC  │ │ GIMP │ │LibreOffice│   │      │
│      │  └──────┘ └──────┘ └──────┘ └──────────┘   │      │
│      └──────────────────────────────────────────────┘      │
│                          │                                  │
│                          ▼                                  │
│      ┌──────────────────────────────────────────────┐      │
│      │              Xvfb :0 (1920x1080)             │      │
│      └──────────────────────────────────────────────┘      │
│                          │                                  │
│                          ▼                                  │
│      ┌──────────────────────────────────────────────┐      │
│      │           Mock logind D-Bus Service          │      │
│      │        (replaces systemd-logind)             │      │
│      └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Why No Privileged Mode?

This image uses a **mock logind D-Bus service** instead of systemd to satisfy GNOME Shell's session management requirements. This approach:

1. **NVCF Compatible** - Works in NVIDIA Cloud Functions and other restricted environments
2. **No Special Capabilities** - Doesn't require `--privileged`, `SYS_ADMIN`, or `NET_ADMIN`
3. **Secure** - Runs with minimal container permissions
4. **Simpler Deployment** - Just `docker run` with port mapping and shared memory

The mock logind service provides the `org.freedesktop.login1` D-Bus interface that GNOME Shell expects, allowing the desktop to run normally without actual systemd.

## Container Requirements

The container only requires:
- `--shm-size=2g` - Shared memory for Chrome (prevents crashes)

Optional:
- Volume mounts for persistent data

## Troubleshooting

### Container not starting
Check the container logs:
```bash
docker logs osworld
```

### Black screen in VNC
Wait a few seconds for GNOME Shell to fully initialize. The startup process includes:
1. D-Bus daemon startup
2. Mock logind service
3. Xvfb virtual display
4. GNOME Shell
5. VNC/noVNC services

### Services not running
Check if processes are running:
```bash
docker exec osworld ps aux | grep -E "(gnome|python|x11vnc|caddy)"
```

## Comparison with Original OSWorld VM

| Feature | Original QEMU VM | Docker Container |
|---------|------------------|------------------|
| Base OS | Ubuntu 22.04.3 LTS | Ubuntu 22.04 LTS |
| Desktop | GNOME Shell 42.9 | GNOME Shell 42.9 |
| Init System | systemd | entrypoint script + mock logind |
| Privileged Mode | N/A (VM) | Not required |
| Hardware | QEMU/KVM | Docker/containerd |

## License

This project is for research and educational purposes.
