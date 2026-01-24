# OSWorld Linux Docker Environment

This Docker image replicates the OSWorld Linux environment (Ubuntu 22.04 with GNOME desktop) for benchmark tasks. It includes all required applications and configurations as specified in the OSWorld setup documentation.

## Features

- **Base OS**: Ubuntu 22.04 LTS with GNOME desktop environment
- **Display**: 1920x1080 resolution via Xvfb (headless)
- **VNC Access**: x11vnc + noVNC for remote desktop access
- **OSWorld Server**: Flask-based API server for automation

### Installed Applications

| Application | Version | Purpose |
|-------------|---------|---------|
| Google Chrome | Latest stable | Web browser tasks (46 benchmark tasks) |
| Chromium | Latest | Alternative for ARM systems |
| LibreOffice | 7.3.7.2 | Office suite tasks |
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
docker build -t osworld-linux .
```

### Run with Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Run with Docker

```bash
docker run -d \
  --name osworld-linux \
  --privileged \
  --shm-size=2g \
  -p 5000:5000 \
  -p 5900:5900 \
  -p 5910:5910 \
  -p 9222:9222 \
  -p 8080:8080 \
  osworld-linux
```

## Accessing the Environment

### Web-based VNC (noVNC)
Open your browser and navigate to:
```
http://localhost:5910/vnc.html
```

### Direct VNC Connection
Use any VNC client to connect to:
```
localhost:5900
```

### OSWorld API Server
The REST API server is available at:
```
http://localhost:5000
```

## Port Configuration

| Port | Service | Description |
|------|---------|-------------|
| 5000 | OSWorld Server | Main API server (Flask) |
| 5900 | x11vnc | VNC server |
| 5910 | noVNC | Web-based VNC access |
| 9222 | Chrome DevTools | Chrome remote debugging |
| 8080 | VLC HTTP | VLC media player control |

## API Endpoints

### Server Status
```bash
curl http://localhost:5000/version
```

### Screenshot
```bash
curl http://localhost:5000/screenshot --output screenshot.png
```

### Execute Command
```bash
curl -X POST http://localhost:5000/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "ls -la", "shell": true}'
```

### Get Accessibility Tree
```bash
curl http://localhost:5000/accessibility
```

### Launch Application
```bash
curl -X POST http://localhost:5000/setup/launch \
  -H "Content-Type: application/json" \
  -d '{"command": ["google-chrome", "--remote-debugging-port=9222"]}'
```

## Verification

Run the verification script to ensure all components are working:

```bash
chmod +x verify.sh
./verify.sh
```

Or run inside the container:
```bash
docker exec osworld-linux /usr/local/bin/verify-apps.sh
```

## Credentials

- **Username**: `user`
- **Password**: `password`

## Configuration Details

### Chrome Configuration
- Remote debugging enabled on port 9222
- Password manager disabled
- Autofill disabled
- Sync disabled

### VLC Configuration
- HTTP interface enabled
- HTTP password: `password`
- HTTP port: 8080

### VS Code Configuration
- Workspace trust disabled
- Telemetry disabled

### LibreOffice Configuration
- Default save formats set to Microsoft Office formats (.docx, .xlsx, .pptx)

### Thunderbird Configuration
- Accessibility tree enabled via: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`

## Comparison with QEMU-based OSWorld

This Docker image provides a similar environment to the QEMU-based OSWorld but with some differences:

| Feature | Docker | QEMU |
|---------|--------|------|
| Startup time | Fast (~10s) | Slower (~60s) |
| Resource usage | Lower | Higher |
| Nested virtualization | Not required | Required (KVM) |
| Snapshot support | Via Docker commits | Native QEMU snapshots |
| Network isolation | Docker networking | QEMU user networking |

## Troubleshooting

### VNC not connecting
```bash
# Check if Xvfb is running
docker exec osworld-linux pgrep Xvfb

# Check if x11vnc is running
docker exec osworld-linux pgrep x11vnc

# Restart services
docker exec osworld-linux /usr/local/bin/entrypoint.sh
```

### OSWorld server not responding
```bash
# Check server logs
docker exec osworld-linux cat /home/user/server/server.log

# Restart server
docker exec osworld-linux pkill -f "python3 main.py"
docker exec osworld-linux su - user -c "cd /home/user/server && python3 main.py &"
```

### Applications not launching
```bash
# Check DISPLAY variable
docker exec osworld-linux echo $DISPLAY

# Test X server
docker exec -u user osworld-linux xdotool getmouselocation
```

## Building for ARM (Apple Silicon)

For ARM-based systems, modify the Dockerfile:
1. Replace Google Chrome with Chromium
2. Use ARM-compatible LibreOffice packages

```dockerfile
# Replace Chrome installation with:
RUN apt-get update && apt-get install -y chromium-browser
```

## License

This Docker configuration is provided as-is for OSWorld benchmark purposes.
