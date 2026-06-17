# Runtime for CUA-Gym web rollouts: Python + Node + Chromium + agent-browser.
# Serves each mock web app via Vite and drives it through Hermes' browser toolset.
# Hermes itself ([web] extras) is installed into an isolated venv at task time.
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ARG POLAR_CUA_IMAGE_VERSION=1
LABEL io.polar.cua-gym-image-version=${POLAR_CUA_IMAGE_VERSION}

COPY --from=node /usr/local/ /usr/local/

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.local \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PATH=/home/polar/.local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash build-essential ca-certificates chromium curl \
        fonts-liberation git python-is-python3 sudo tmux wget xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# requests is used by the CUA initial_setup.py / reward.py scripts (run as the
# system python3), so install it system-wide (visible despite PYTHONNOUSERSITE).
RUN pip install --no-cache-dir requests

RUN useradd -m -s /bin/bash polar \
    && echo 'polar ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /polar/session /home/polar/.local/bin \
    && chown -R polar:polar /polar/session /home/polar

USER polar
WORKDIR /polar/session

RUN npm install -g agent-browser@0.27.2 && agent-browser --version
