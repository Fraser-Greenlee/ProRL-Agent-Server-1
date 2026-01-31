# OSWorld Linux Environment Docker Image - GNOME Shell (No Privileged Mode)
# Replicates Ubuntu 22.04 with GNOME Shell desktop (same as original OSWorld QEMU VM)
# Uses a mock logind D-Bus service instead of systemd for NVCF compatibility
#
# Build: docker build -t osworld-linux .
# Run:   docker run -d --name osworld --shm-size=2g -p 8000:8000 -p 9222:9222 osworld-linux

FROM ubuntu:22.04

LABEL maintainer="OSWorld Docker"
LABEL description="OSWorld Linux Environment with Ubuntu 22.04 GNOME Shell Desktop"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

ENV USER_NAME=user
ENV USER_PASSWORD=password
ENV USER_UID=1000
ENV USER_GID=1000

ENV DISPLAY=:0
ENV RESOLUTION=1920x1080
ENV COLOR_DEPTH=24

# ============================================
# Base System
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    wget \
    gnupg \
    software-properties-common \
    lsb-release \
    sudo \
    locales \
    tzdata \
    dbus \
    dbus-x11 \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    python3-dbus \
    gir1.2-gtk-3.0 \
    xvfb \
    x11vnc \
    xdotool \
    xclip \
    xsel \
    xterm \
    wmctrl \
    ffmpeg \
    socat \
    net-tools \
    iputils-ping \
    at-spi2-core \
    libatspi2.0-0 \
    python3-pyatspi \
    libx11-dev \
    libxfixes-dev \
    libxcursor-dev \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto \
    fonts-noto-cjk \
    fontconfig \
    git \
    unzip \
    zip \
    python3-lxml \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ============================================
# Install Ubuntu GNOME Desktop
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnome-shell \
    gnome-session \
    ubuntu-session \
    gnome-terminal \
    gnome-screenshot \
    nautilus \
    gedit \
    gvfs \
    gvfs-backends \
    dconf-cli \
    dconf-editor \
    eog \
    file-roller \
    gnome-shell-extension-ubuntu-dock \
    gnome-shell-extension-appindicator \
    gnome-shell-extension-desktop-icons-ng \
    mutter \
    gnome-settings-daemon \
    gnome-control-center \
    adwaita-icon-theme \
    adwaita-icon-theme-full \
    yaru-theme-gnome-shell \
    yaru-theme-gtk \
    yaru-theme-icon \
    yaru-theme-sound \
    yelp \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install noVNC and Caddy
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    debian-keyring debian-archive-keyring apt-transport-https \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list \
    && apt-get update \
    && apt-get install -y caddy \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install Applications
# ============================================

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    libreoffice-gtk3 \
    vlc \
    vlc-plugin-access-extra \
    gimp \
    gimp-data \
    thunderbird \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list \
    && apt-get update \
    && apt-get install -y code \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Create user
# ============================================

RUN groupadd -g ${USER_GID} ${USER_NAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd \
    && usermod -aG sudo,audio,video ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ============================================
# Create OSWorld server
# ============================================

RUN mkdir -p /home/${USER_NAME}/server
COPY server/ /home/${USER_NAME}/server/

RUN pip3 install flask pyautogui pillow mss pynput psutil requests lxml dbus-python

# ============================================
# Create Mock logind D-Bus service
# ============================================

RUN cat > /usr/local/bin/mock-logind.py << 'MOCKLOGIND'
#!/usr/bin/env python3
"""
Mock logind D-Bus service for running GNOME Shell without systemd.
Provides minimal org.freedesktop.login1 interface.
"""

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import os

class MockLogindManager(dbus.service.Object):
    def __init__(self, bus):
        self.bus = bus
        self.bus_name = dbus.service.BusName('org.freedesktop.login1', bus)
        dbus.service.Object.__init__(self, bus, '/org/freedesktop/login1')

        # Create a session object
        self.session = MockLogindSession(bus, '/org/freedesktop/login1/session/auto')
        self.seat = MockLogindSeat(bus, '/org/freedesktop/login1/seat/seat0')
        self.user = MockLogindUser(bus, '/org/freedesktop/login1/user/_1000')

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='', out_signature='a(susso)')
    def ListSessions(self):
        return [('auto', 1000, 'user', 'seat0', '/org/freedesktop/login1/session/auto')]

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='', out_signature='a(uso)')
    def ListUsers(self):
        return [(1000, 'user', '/org/freedesktop/login1/user/_1000')]

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='', out_signature='a(so)')
    def ListSeats(self):
        return [('seat0', '/org/freedesktop/login1/seat/seat0')]

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='s', out_signature='o')
    def GetSession(self, session_id):
        return '/org/freedesktop/login1/session/auto'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='u', out_signature='o')
    def GetUser(self, uid):
        return '/org/freedesktop/login1/user/_1000'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='s', out_signature='o')
    def GetSeat(self, seat_id):
        return '/org/freedesktop/login1/seat/seat0'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='', out_signature='o')
    def GetSessionByPID(self, pid=0):
        return '/org/freedesktop/login1/session/auto'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='u', out_signature='o')
    def GetUserByPID(self, pid):
        return '/org/freedesktop/login1/user/_1000'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='ssss', out_signature='h')
    def Inhibit(self, what, who, why, mode):
        return os.open('/dev/null', os.O_RDONLY)

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='b', out_signature='')
    def PowerOff(self, interactive):
        pass

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='b', out_signature='')
    def Reboot(self, interactive):
        pass

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='b', out_signature='')
    def Suspend(self, interactive):
        pass

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='s', out_signature='s')
    def CanPowerOff(self, action=''):
        return 'yes'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='s', out_signature='s')
    def CanReboot(self, action=''):
        return 'yes'

    @dbus.service.method('org.freedesktop.login1.Manager',
                         in_signature='s', out_signature='s')
    def CanSuspend(self, action=''):
        return 'yes'

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = {
            'IdleHint': False,
            'IdleSinceHint': dbus.UInt64(0),
            'IdleSinceHintMonotonic': dbus.UInt64(0),
            'PreparingForShutdown': False,
            'PreparingForSleep': False,
            'Docked': True,
            'LidClosed': False,
            'OnExternalPower': True,
        }
        return props.get(prop, '')

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        return {
            'IdleHint': False,
            'IdleSinceHint': dbus.UInt64(0),
            'IdleSinceHintMonotonic': dbus.UInt64(0),
            'PreparingForShutdown': False,
            'PreparingForSleep': False,
            'Docked': True,
            'LidClosed': False,
            'OnExternalPower': True,
        }


class MockLogindSession(dbus.service.Object):
    def __init__(self, bus, path):
        dbus.service.Object.__init__(self, bus, path)

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='', out_signature='')
    def Activate(self):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='', out_signature='')
    def Lock(self):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='', out_signature='')
    def Unlock(self):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='b', out_signature='')
    def SetIdleHint(self, idle):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='s', out_signature='')
    def SetType(self, type_):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='uu', out_signature='')
    def TakeControl(self, force=0, uid=0):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='', out_signature='')
    def ReleaseControl(self):
        pass

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='uu', out_signature='hb')
    def TakeDevice(self, major, minor):
        # Return a file descriptor and whether it's paused
        fd = os.open('/dev/null', os.O_RDWR)
        return (fd, False)

    @dbus.service.method('org.freedesktop.login1.Session',
                         in_signature='uu', out_signature='')
    def ReleaseDevice(self, major, minor):
        pass

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = {
            'Id': 'auto',
            'User': (dbus.UInt32(1000), '/org/freedesktop/login1/user/_1000'),
            'Name': 'user',
            'Timestamp': dbus.UInt64(0),
            'TimestampMonotonic': dbus.UInt64(0),
            'VTNr': dbus.UInt32(1),
            'Seat': ('seat0', '/org/freedesktop/login1/seat/seat0'),
            'TTY': '',
            'Display': ':0',
            'Remote': False,
            'RemoteHost': '',
            'RemoteUser': '',
            'Service': '',
            'Desktop': 'ubuntu',
            'Scope': 'session-auto.scope',
            'Leader': dbus.UInt32(os.getpid()),
            'Audit': dbus.UInt32(0),
            'Type': 'x11',
            'Class': 'user',
            'Active': True,
            'State': 'active',
            'IdleHint': False,
            'IdleSinceHint': dbus.UInt64(0),
            'IdleSinceHintMonotonic': dbus.UInt64(0),
            'LockedHint': False,
        }
        return props.get(prop, '')

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        return {
            'Id': 'auto',
            'User': (dbus.UInt32(1000), '/org/freedesktop/login1/user/_1000'),
            'Name': 'user',
            'Active': True,
            'State': 'active',
            'Type': 'x11',
            'Class': 'user',
            'Display': ':0',
            'Seat': ('seat0', '/org/freedesktop/login1/seat/seat0'),
        }


class MockLogindSeat(dbus.service.Object):
    def __init__(self, bus, path):
        dbus.service.Object.__init__(self, bus, path)

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = {
            'Id': 'seat0',
            'ActiveSession': ('auto', '/org/freedesktop/login1/session/auto'),
            'CanGraphical': True,
            'CanMultiSession': True,
            'CanTTY': True,
            'Sessions': [('auto', '/org/freedesktop/login1/session/auto')],
        }
        return props.get(prop, '')

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        return {
            'Id': 'seat0',
            'ActiveSession': ('auto', '/org/freedesktop/login1/session/auto'),
            'CanGraphical': True,
        }


class MockLogindUser(dbus.service.Object):
    def __init__(self, bus, path):
        dbus.service.Object.__init__(self, bus, path)

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        props = {
            'UID': dbus.UInt32(1000),
            'GID': dbus.UInt32(1000),
            'Name': 'user',
            'Timestamp': dbus.UInt64(0),
            'TimestampMonotonic': dbus.UInt64(0),
            'RuntimePath': '/run/user/1000',
            'Slice': 'user-1000.slice',
            'Display': ('auto', '/org/freedesktop/login1/session/auto'),
            'State': 'active',
            'Sessions': [('auto', '/org/freedesktop/login1/session/auto')],
            'IdleHint': False,
            'IdleSinceHint': dbus.UInt64(0),
            'IdleSinceHintMonotonic': dbus.UInt64(0),
            'Linger': False,
        }
        return props.get(prop, '')

    @dbus.service.method(dbus.PROPERTIES_IFACE,
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        return {
            'UID': dbus.UInt32(1000),
            'Name': 'user',
            'State': 'active',
        }


if __name__ == '__main__':
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    manager = MockLogindManager(bus)
    print("Mock logind service started")
    loop = GLib.MainLoop()
    loop.run()
MOCKLOGIND
RUN chmod +x /usr/local/bin/mock-logind.py

# ============================================
# Configure Chrome
# ============================================

RUN cat > /usr/local/bin/google-chrome-wrapper << 'CHROMEWRAP'
#!/bin/bash
CHROME_DIR="/home/user/.config/google-chrome-debug"
if [ -f "$CHROME_DIR/SingletonLock" ]; then
    LOCK_TARGET=$(readlink "$CHROME_DIR/SingletonLock" 2>/dev/null)
    if [ -n "$LOCK_TARGET" ]; then
        PID=$(echo "$LOCK_TARGET" | cut -d'-' -f2)
        if ! kill -0 "$PID" 2>/dev/null; then
            rm -f "$CHROME_DIR/SingletonLock" "$CHROME_DIR/SingletonSocket" "$CHROME_DIR/SingletonCookie"
        fi
    fi
fi
exec /usr/bin/google-chrome-stable --no-sandbox --remote-debugging-port=9223 --remote-debugging-address=127.0.0.1 --user-data-dir=/home/user/.config/google-chrome-debug --no-first-run --no-default-browser-check "$@"
CHROMEWRAP
RUN chmod +x /usr/local/bin/google-chrome-wrapper

RUN cat > /usr/share/applications/google-chrome.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=Google Chrome
Exec=/usr/local/bin/google-chrome-wrapper %U
StartupNotify=true
Terminal=false
Icon=google-chrome
Type=Application
Categories=Network;WebBrowser;
EOF

RUN mkdir -p /home/${USER_NAME}/.config/google-chrome-debug

# ============================================
# Configure gnome-terminal (fix for Docker)
# ============================================

# gnome-terminal needs special handling in Docker without systemd
# Create a wrapper that launches it properly
# Create wrapper for gnome-terminal-server with proper locale
RUN echo '#!/bin/bash' > /usr/local/bin/gnome-terminal-server-wrapper \
    && echo 'export LANG=en_US.UTF-8' >> /usr/local/bin/gnome-terminal-server-wrapper \
    && echo 'export LC_ALL=en_US.UTF-8' >> /usr/local/bin/gnome-terminal-server-wrapper \
    && echo 'export DISPLAY=${DISPLAY:-:0}' >> /usr/local/bin/gnome-terminal-server-wrapper \
    && echo 'exec /usr/libexec/gnome-terminal-server "$@"' >> /usr/local/bin/gnome-terminal-server-wrapper \
    && chmod +x /usr/local/bin/gnome-terminal-server-wrapper

# Fix D-Bus service file to use the wrapper
RUN cat > /usr/share/dbus-1/services/org.gnome.Terminal.service << 'EOF'
[D-BUS Service]
Name=org.gnome.Terminal
Exec=/usr/local/bin/gnome-terminal-server-wrapper
EOF

# ============================================
# Configure other apps
# ============================================

RUN mkdir -p /home/${USER_NAME}/.config/vlc \
    && cat > /home/${USER_NAME}/.config/vlc/vlcrc << 'EOF'
[http]
http-host=0.0.0.0
http-port=8100
http-password=

[qt]
qt-privacy-ask=0
EOF

RUN mkdir -p /home/${USER_NAME}/.config/Code/User \
    && cat > /home/${USER_NAME}/.config/Code/User/settings.json << 'EOF'
{
    "window.zoomLevel": 0,
    "editor.fontSize": 14,
    "telemetry.telemetryLevel": "off",
    "update.mode": "none"
}
EOF

RUN mkdir -p /home/${USER_NAME}/.config/libreoffice/4/user \
    && cat > /home/${USER_NAME}/.config/libreoffice/4/user/registrymodifications.xcu << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>
</oor:items>
EOF

# ============================================
# Disable GNOME Keyring password prompt
# ============================================

# Disable gnome-keyring D-Bus activation (this is the key fix)
RUN mv /usr/share/dbus-1/services/org.gnome.keyring.service /usr/share/dbus-1/services/org.gnome.keyring.service.disabled 2>/dev/null || true \
    && mv /usr/share/dbus-1/services/org.freedesktop.secrets.service /usr/share/dbus-1/services/org.freedesktop.secrets.service.disabled 2>/dev/null || true

# Create keyring directory with login keyring (empty password)
RUN mkdir -p /home/${USER_NAME}/.local/share/keyrings \
    && cat > /home/${USER_NAME}/.local/share/keyrings/login.keyring << 'EOF'
[keyring]
display-name=Login
ctime=1234567890
mtime=1234567890
lock-on-idle=false
lock-after=false
EOF

RUN echo "login" > /home/${USER_NAME}/.local/share/keyrings/default

# Disable keyring daemon autostart for secrets component
RUN mkdir -p /home/${USER_NAME}/.config/autostart \
    && cat > /home/${USER_NAME}/.config/autostart/gnome-keyring-secrets.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Secret Storage Service
Exec=/bin/true
Hidden=true
NoDisplay=true
X-GNOME-Autostart-enabled=false
EOF

# ============================================
# Configure Caddy
# ============================================

RUN cat > /etc/caddy/Caddyfile << 'CADDYFILE'
:8000 {
    @root path /
    @vnchtml path /vnc.html

    redir @root /vnc/vnc.html 302
    redir @vnchtml /vnc/vnc.html 302

    handle /vnc/* {
        uri strip_prefix /vnc
        reverse_proxy localhost:5910
    }

    handle /websockify {
        reverse_proxy localhost:5910
    }

    handle /api/* {
        uri strip_prefix /api
        reverse_proxy localhost:5000
    }

    handle /vlc/* {
        uri strip_prefix /vlc
        reverse_proxy localhost:8100
    }

    handle /chrome/* {
        uri strip_prefix /chrome
        reverse_proxy localhost:9222
    }
}
CADDYFILE

# ============================================
# Configure GNOME Shell settings via dconf
# ============================================

RUN mkdir -p /etc/dconf/profile && \
    echo 'user-db:user' > /etc/dconf/profile/user && \
    echo 'system-db:local' >> /etc/dconf/profile/user

RUN mkdir -p /etc/dconf/db/local.d

RUN cat > /etc/dconf/db/local.d/00-osworld << 'DCONF_EOF'
[org/gnome/shell]
favorite-apps=['google-chrome.desktop', 'thunderbird.desktop', 'code.desktop', 'vlc.desktop', 'libreoffice-writer.desktop', 'libreoffice-calc.desktop', 'libreoffice-impress.desktop', 'gimp.desktop', 'org.gnome.Nautilus.desktop', 'org.gnome.Terminal.desktop', 'yelp.desktop']
enabled-extensions=['ubuntu-dock@ubuntu.com', 'ubuntu-appindicators@ubuntu.com', 'ding@rastersoft.com']

[org/gnome/desktop/background]
picture-uri='file:///usr/share/backgrounds/warty-final-ubuntu.png'
picture-uri-dark='file:///usr/share/backgrounds/warty-final-ubuntu.png'

[org/gnome/desktop/interface]
gtk-theme='Yaru'
icon-theme='Yaru'
cursor-theme='Yaru'

[org/gnome/desktop/wm/preferences]
button-layout=':minimize,maximize,close'

[org/gnome/desktop/screensaver]
lock-enabled=false
idle-activation-enabled=false

[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
sleep-inactive-battery-type='nothing'
DCONF_EOF

RUN dconf update

# ============================================
# Create entrypoint script
# ============================================

RUN cat > /usr/local/bin/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=ubuntu:GNOME
export GNOME_SHELL_SESSION_MODE=ubuntu

# Create runtime directories
mkdir -p /run/user/1000
chmod 700 /run/user/1000
chown user:user /run/user/1000

mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

mkdir -p /run/dbus
chmod 755 /run/dbus

echo "Starting D-Bus system daemon..."
rm -f /run/dbus/pid
dbus-daemon --system --fork
sleep 1

echo "Starting mock logind service..."
python3 /usr/local/bin/mock-logind.py &
sleep 2

echo "Starting Xvfb..."
Xvfb :0 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2

# Start dbus session for user
echo "Starting D-Bus session..."
su - user -c "mkdir -p /run/user/1000 && dbus-daemon --session --address=unix:path=/run/user/1000/bus --fork" 2>/dev/null || true
sleep 1

echo "Starting GNOME Shell..."

# Start gnome-settings-daemon
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && /usr/libexec/gsd-xsettings &" 2>/dev/null || true
sleep 1

# Start GNOME Shell
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && export XDG_SESSION_TYPE=x11 && export XDG_CURRENT_DESKTOP=ubuntu:GNOME && export GNOME_SHELL_SESSION_MODE=ubuntu && gnome-shell --x11 2>&1 | head -50 &" 2>/dev/null || true
sleep 5

# Wait for GNOME Shell to be ready
for i in $(seq 1 30); do
    if pgrep -x gnome-shell > /dev/null; then
        echo "GNOME Shell started successfully"
        break
    fi
    echo "Waiting for GNOME Shell... ($i)"
    sleep 1
done

# Start gvfsd (required for Nautilus file manager)
echo "Starting gvfsd..."
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && /usr/libexec/gvfsd &" 2>/dev/null || true
sleep 1

# Start x11vnc
echo "Starting x11vnc..."
su - user -c "x11vnc -display :0 -rfbport 5900 -forever -shared -nopw -xkb -noshm -noxdamage &"
sleep 2

# Start noVNC
echo "Starting noVNC..."
su - user -c "websockify --web=/usr/share/novnc 5910 localhost:5900 &"
sleep 1

# Start Chrome debug port forwarder
echo "Starting Chrome debug port forwarder..."
socat TCP-LISTEN:9222,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:9223 &
sleep 1

# Start OSWorld server
echo "Starting OSWorld server..."
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && cd /home/user/server && python3 main.py" &
sleep 2

# Start Caddy
echo "Starting Caddy..."
caddy start --config /etc/caddy/Caddyfile &
sleep 1

echo ""
echo "============================================"
echo "OSWorld Docker Container Started"
echo "============================================"
echo "Access: http://localhost:8000/vnc.html"
echo "============================================"

# Keep container running
tail -f /dev/null
ENTRYPOINT_EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

# ============================================
# Set ownership
# ============================================

RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

EXPOSE 8000 5000 5900 5910 9222 8100

WORKDIR /home/${USER_NAME}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
