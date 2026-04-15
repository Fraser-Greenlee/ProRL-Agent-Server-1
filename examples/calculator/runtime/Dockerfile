FROM node:22-bookworm-slim

ARG POLAR_CALCULATOR_IMAGE_VERSION=2

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.local \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PATH=/home/polar/.local/bin:/home/polar/.venv/bin:${PATH}

LABEL io.polar.calculator-image-version="${POLAR_CALCULATOR_IMAGE_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash polar \
    && mkdir -p /polar/session /home/polar/.local /home/polar/.venv \
    && chown -R polar:polar /polar/session /home/polar

USER polar
WORKDIR /polar/session
