# Runtime for LiteResearcher rollouts: Python + the MCP server's deps.
# Hermes itself is installed into an isolated venv at task time (prepare step).
ARG POLAR_LITE_IMAGE_VERSION=1
FROM python:3.12-slim-bookworm

ARG POLAR_LITE_IMAGE_VERSION=1
LABEL io.polar.lite-researcher-image-version=${POLAR_LITE_IMAGE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    PATH=/home/polar/.local/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Installed into the system interpreter so the MCP server (run as `python3`)
# imports them regardless of PYTHONNOUSERSITE.
RUN pip install --no-cache-dir mcp httpx

RUN useradd -m -s /bin/bash polar \
    && mkdir -p /polar/session /home/polar/.local/bin \
    && chown -R polar:polar /polar/session /home/polar

USER polar
WORKDIR /polar/session
