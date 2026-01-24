# OSWorld Linux Environment Docker Image
# Replicates Ubuntu 22.04 with Unity desktop and all required applications
# for OSWorld benchmark tasks
#
# Build: docker build -t osworld-linux .
# Run:   docker run -d --name osworld --privileged --shm-size=2g -p 5000:5000 -p 5900:5900 -p 5910:5910 osworld-linux

FROM ubuntu:22.04

LABEL maintainer="OSWorld Docker"
LABEL description="OSWorld Linux Environment with Ubuntu 22.04 Unity Desktop"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Default user credentials (same as OSWorld)
ENV USER_NAME=user
ENV USER_PASSWORD=password
ENV USER_UID=1000
ENV USER_GID=1000

# Display settings
ENV DISPLAY=:0
ENV RESOLUTION=1920x1080
ENV COLOR_DEPTH=24

# Port configuration
ENV OSWORLD_SERVER_PORT=5000
ENV VNC_PORT=5900
ENV NOVNC_PORT=5910
ENV CHROME_DEBUG_PORT=9222
ENV VLC_HTTP_PORT=8080

# ============================================
# STAGE 1: Base System
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
    gir1.2-gtk-3.0 \
    xvfb \
    x11vnc \
    xdotool \
    xclip \
    xsel \
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
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ============================================
# STAGE 2: Install Ubuntu Unity Desktop Environment
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    ubuntu-unity-desktop \
    gnome-terminal \
    gnome-screenshot \
    gnome-software \
    nautilus \
    gedit \
    gvfs \
    gvfs-backends \
    dconf-cli \
    dconf-editor \
    eog \
    file-roller \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 3: Install noVNC
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 4: Install Google Chrome
# ============================================

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 5: Install LibreOffice 7.3.7.2
# ============================================

RUN apt-get update && apt-get remove -y libreoffice* || true

RUN mkdir -p /tmp/libreoffice && cd /tmp/libreoffice \
    && wget -q "https://downloadarchive.documentfoundation.org/libreoffice/old/7.3.7.2/deb/x86_64/LibreOffice_7.3.7.2_Linux_x86-64_deb.tar.gz" \
    && tar xzf LibreOffice_7.3.7.2_Linux_x86-64_deb.tar.gz \
    && cd LibreOffice_7.3.7.2*_Linux_x86-64_deb/DEBS \
    && dpkg -i *.deb || apt-get install -f -y \
    && cd / && rm -rf /tmp/libreoffice \
    && ln -sf /opt/libreoffice7.3/program/soffice /usr/local/bin/libreoffice \
    && ln -sf /opt/libreoffice7.3/program/soffice /usr/local/bin/soffice \
    && ln -sf /opt/libreoffice7.3/program/soffice /usr/local/bin/lowriter \
    && ln -sf /opt/libreoffice7.3/program/soffice /usr/local/bin/localc \
    && ln -sf /opt/libreoffice7.3/program/soffice /usr/local/bin/loimpress

# ============================================
# STAGE 6: Install GIMP
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends gimp && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 7: Install VLC Media Player
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    vlc vlc-plugin-access-extra && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 8: Install Visual Studio Code
# ============================================

RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /tmp/packages.microsoft.gpg \
    && install -D -o root -g root -m 644 /tmp/packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list \
    && apt-get update \
    && apt-get install -y code \
    && rm -rf /var/lib/apt/lists/* /tmp/packages.microsoft.gpg

# ============================================
# STAGE 9: Install Thunderbird
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends thunderbird && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 10: Install Python Dependencies
# ============================================

RUN pip3 install --no-cache-dir \
    python3-xlib==0.15 \
    PyAutoGUI==0.9.54 \
    Pillow==10.1.0 \
    pynput \
    requests \
    flask \
    numpy \
    lxml \
    pygame

# ============================================
# STAGE 11: Install Fonts for LibreOffice
# ============================================

RUN mkdir -p /tmp/fonts && cd /tmp/fonts \
    && wget -q "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/fonts_20250608_fixed.zip" -O fonts.zip 2>/dev/null || true \
    && if [ -f fonts.zip ]; then \
        unzip -q fonts.zip -d /usr/share/fonts/ || true; \
        fc-cache -fv; \
       fi \
    && cd / && rm -rf /tmp/fonts

# ============================================
# STAGE 12: Create User Account
# ============================================

RUN groupadd -g ${USER_GID} ${USER_NAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd \
    && usermod -aG sudo ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/${USER_NAME}/server \
    && mkdir -p /home/${USER_NAME}/Desktop \
    && mkdir -p /home/${USER_NAME}/Documents \
    && mkdir -p /home/${USER_NAME}/Downloads \
    && mkdir -p /home/${USER_NAME}/Pictures \
    && mkdir -p /home/${USER_NAME}/Videos \
    && mkdir -p /home/${USER_NAME}/.config \
    && mkdir -p /home/${USER_NAME}/.local/share/applications \
    && mkdir -p /run/user/${USER_UID} \
    && chmod 700 /run/user/${USER_UID} \
    && chown ${USER_NAME}:${USER_NAME} /run/user/${USER_UID}

# ============================================
# STAGE 13: Copy OSWorld Server Files
# ============================================

COPY server/main.py /home/${USER_NAME}/server/main.py
COPY server/pyxcursor.py /home/${USER_NAME}/server/pyxcursor.py
RUN mkdir -p /home/${USER_NAME}/server/screenshots

# ============================================
# STAGE 14: Create Desktop Shortcut for Home
# ============================================

RUN mkdir -p /home/${USER_NAME}/Desktop \
    && cat > /home/${USER_NAME}/Desktop/Home.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Home
Icon=user-home
Exec=nautilus /home/user
Terminal=false
EOF
RUN chmod +x /home/${USER_NAME}/Desktop/Home.desktop

# ============================================
# STAGE 15: Configure Unity Compiz Settings
# ============================================

# Create compizconfig directory and configuration file
RUN mkdir -p /home/${USER_NAME}/.config/compiz-1/compizconfig \
    && cat > /home/${USER_NAME}/.config/compiz-1/compizconfig/config << 'EOF'
[general]
profile = unity
backend = gsettings
integration = true
plugin_list_autosort = true

[general_ubuntu]
profile = unity

[general_unity]
backend = gsettings
profile = unity
integration = true
plugin_list_autosort = true
EOF

# Create script to configure Unity settings
RUN cat > /usr/local/bin/configure-unity.sh << 'UNITYSCRIPT'
#!/bin/bash
export DISPLAY=:0
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XDG_RUNTIME_DIR=/run/user/1000

# Wait for Unity to be ready
sleep 5

# Set compiz profile to unity
gsettings set org.compiz current-profile 'unity' 2>/dev/null || true

# Configure Unity Launcher favorites (matching OSWorld QEMU VM)
# Order: Chrome, Thunderbird, VS Code, VLC, Writer, Calc, Impress, GIMP, Terminal, Software, Help, Settings
gsettings set com.canonical.Unity.Launcher favorites "['application://google-chrome.desktop', 'application://thunderbird.desktop', 'application://code.desktop', 'application://vlc.desktop', 'application://libreoffice7.3-writer.desktop', 'application://libreoffice7.3-calc.desktop', 'application://libreoffice7.3-impress.desktop', 'application://gimp.desktop', 'application://org.gnome.Terminal.desktop', 'application://org.gnome.Software.desktop', 'application://yelp.desktop', 'application://unity-control-center.desktop', 'unity://running-apps', 'unity://expo-icon', 'unity://devices']" 2>/dev/null || true

# Set wallpaper to Ubuntu jellyfish
gsettings set org.gnome.desktop.background picture-uri 'file:///usr/share/backgrounds/warty-final-ubuntu.png' 2>/dev/null || true
gsettings set org.gnome.desktop.background picture-uri-dark 'file:///usr/share/backgrounds/warty-final-ubuntu.png' 2>/dev/null || true

# Enable accessibility for Thunderbird
gsettings set org.gnome.desktop.interface toolkit-accessibility true 2>/dev/null || true

# Set theme
gsettings set org.gnome.desktop.interface gtk-theme 'Yaru' 2>/dev/null || true
gsettings set org.gnome.desktop.interface icon-theme 'Yaru' 2>/dev/null || true

echo "Unity configuration complete"
UNITYSCRIPT
RUN chmod +x /usr/local/bin/configure-unity.sh

# ============================================
# STAGE 16: Configure Chrome with remote debugging
# ============================================

RUN mkdir -p /etc/opt/chrome/policies/managed \
    && cat > /etc/opt/chrome/policies/managed/osworld.json << 'EOF'
{
  "PasswordManagerEnabled": false,
  "AutofillAddressEnabled": false,
  "AutofillCreditCardEnabled": false,
  "BrowserSignin": 0,
  "SyncDisabled": true,
  "MetricsReportingEnabled": false
}
EOF

# Create Chrome desktop entry with debugging
# Chrome listens on internal port 9223, socat forwards from 0.0.0.0:9222
RUN cat > /usr/share/applications/google-chrome.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=Google Chrome
GenericName=Web Browser
Comment=Access the Internet
Exec=/usr/bin/google-chrome-stable --remote-debugging-port=9223 --user-data-dir=/home/user/.config/google-chrome-debug --no-first-run --no-default-browser-check %U
StartupNotify=true
Terminal=false
Icon=google-chrome
Type=Application
Categories=Network;WebBrowser;
MimeType=application/pdf;application/rdf+xml;application/rss+xml;application/xhtml+xml;application/xhtml_xml;application/xml;image/gif;image/jpeg;image/png;image/webp;text/html;text/xml;x-scheme-handler/http;x-scheme-handler/https;
Actions=new-window;new-private-window;
EOF

# Create Chrome debug data directory
RUN mkdir -p /home/${USER_NAME}/.config/google-chrome-debug

# ============================================
# STAGE 17: Configure VLC
# ============================================

RUN mkdir -p /home/${USER_NAME}/.config/vlc \
    && cat > /home/${USER_NAME}/.config/vlc/vlcrc << 'EOF'
[http]
http-host=0.0.0.0
http-port=8080
http-password=password

[core]
extraintf=http
EOF

# ============================================
# STAGE 18: Configure VS Code
# ============================================

RUN mkdir -p /home/${USER_NAME}/.config/Code/User \
    && cat > /home/${USER_NAME}/.config/Code/User/settings.json << 'EOF'
{
    "security.workspace.trust.enabled": false,
    "security.workspace.trust.startupPrompt": "never",
    "update.mode": "none",
    "telemetry.enableTelemetry": false,
    "telemetry.enableCrashReporter": false
}
EOF

# ============================================
# STAGE 19: Configure LibreOffice
# ============================================

RUN mkdir -p /home/${USER_NAME}/.config/libreoffice/4/user \
    && cat > /home/${USER_NAME}/.config/libreoffice/4/user/registrymodifications.xcu << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Setup/Office/Factories/org.openoffice.Setup:Factory['com.sun.star.text.TextDocument']"><prop oor:name="ooSetupFactoryDefaultFilter" oor:op="fuse"><value>MS Word 2007 XML</value></prop></item>
<item oor:path="/org.openoffice.Setup/Office/Factories/org.openoffice.Setup:Factory['com.sun.star.sheet.SpreadsheetDocument']"><prop oor:name="ooSetupFactoryDefaultFilter" oor:op="fuse"><value>Calc MS Excel 2007 XML</value></prop></item>
<item oor:path="/org.openoffice.Setup/Office/Factories/org.openoffice.Setup:Factory['com.sun.star.presentation.PresentationDocument']"><prop oor:name="ooSetupFactoryDefaultFilter" oor:op="fuse"><value>Impress MS PowerPoint 2007 XML</value></prop></item>
</oor:items>
EOF

# ============================================
# STAGE 20: Create LibreOffice desktop files
# ============================================

RUN cat > /usr/share/applications/libreoffice-writer.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=LibreOffice Writer
Comment=Word Processor
Exec=/opt/libreoffice7.3/program/soffice --writer %U
Terminal=false
Type=Application
Icon=libreoffice-writer
Categories=Office;WordProcessor;
MimeType=application/vnd.oasis.opendocument.text;
EOF

RUN cat > /usr/share/applications/libreoffice-calc.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=LibreOffice Calc
Comment=Spreadsheet
Exec=/opt/libreoffice7.3/program/soffice --calc %U
Terminal=false
Type=Application
Icon=libreoffice-calc
Categories=Office;Spreadsheet;
MimeType=application/vnd.oasis.opendocument.spreadsheet;
EOF

RUN cat > /usr/share/applications/libreoffice-impress.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Name=LibreOffice Impress
Comment=Presentation
Exec=/opt/libreoffice7.3/program/soffice --impress %U
Terminal=false
Type=Application
Icon=libreoffice-impress
Categories=Office;Presentation;
MimeType=application/vnd.oasis.opendocument.presentation;
EOF

# ============================================
# STAGE 21: Create Entrypoint Script
# ============================================

RUN cat > /usr/local/bin/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

export DISPLAY=:0
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

# Create runtime directories
mkdir -p /run/user/1000
chmod 700 /run/user/1000
chown user:user /run/user/1000

mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

# Start dbus system daemon
mkdir -p /run/dbus
rm -f /run/dbus/pid
dbus-daemon --system --fork 2>/dev/null || true

echo "Starting Xvfb..."
Xvfb :0 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2

# Start dbus session for user
su - user -c "mkdir -p /run/user/1000 && dbus-daemon --session --address=unix:path=/run/user/1000/bus --fork" 2>/dev/null || true
sleep 1

echo "Starting Unity desktop..."

# Start unity-settings-daemon
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && /usr/lib/unity-settings-daemon/unity-settings-daemon &" 2>/dev/null || true
sleep 1

# Start unity-panel-service
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && /usr/lib/x86_64-linux-gnu/unity/unity-panel-service &" 2>/dev/null || true
sleep 1

# Apply Unity configuration first (before starting compiz)
echo "Configuring Unity..."
su - user -c "/usr/local/bin/configure-unity.sh" 2>/dev/null || true
sleep 2

# Start compiz with Unity profile
echo "Starting compiz..."
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && export XDG_CURRENT_DESKTOP=Unity && export COMPIZ_CONFIG_PROFILE=unity && compiz --replace ccp &" 2>/dev/null || true
sleep 10

# Start x11vnc
echo "Starting x11vnc..."
su - user -c "x11vnc -display :0 -rfbport 5900 -forever -shared -nopw -xkb &"
sleep 1

# Start noVNC
echo "Starting noVNC..."
su - user -c "websockify --web=/usr/share/novnc 5910 localhost:5900 &"
sleep 1

# Start socat to forward Chrome debug port from localhost to all interfaces
echo "Starting Chrome debug port forwarder..."
socat TCP-LISTEN:9222,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:9223 &
sleep 1

# Start OSWorld server
echo "Starting OSWorld server..."
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && cd /home/user/server && python3 main.py" &
sleep 2

# Restart compiz after everything is running to ensure Unity shell loads
echo "Restarting compiz to load Unity shell..."
su - user -c "export DISPLAY=:0 && export XDG_RUNTIME_DIR=/run/user/1000 && export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus && export XDG_CURRENT_DESKTOP=Unity && export COMPIZ_CONFIG_PROFILE=unity && pkill compiz; sleep 3; compiz --replace ccp &" 2>/dev/null || true
sleep 10

# Close the Keyboard Shortcuts dialog that appears on first launch
su - user -c "export DISPLAY=:0 && xdotool key Escape" 2>/dev/null || true

echo ""
echo "============================================"
echo "OSWorld Docker Container Started"
echo "============================================"
echo "OSWorld Server: http://localhost:5000"
echo "noVNC Web Access: http://localhost:5910/vnc.html"
echo "VNC Direct: localhost:5900"
echo "============================================"

# Keep container running
tail -f /dev/null
ENTRYPOINT_EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

# ============================================
# STAGE 22: Set Ownership
# ============================================

RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# ============================================
# STAGE 23: Expose Ports
# ============================================

EXPOSE 5000 5900 5910 9222 8080

WORKDIR /home/${USER_NAME}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
