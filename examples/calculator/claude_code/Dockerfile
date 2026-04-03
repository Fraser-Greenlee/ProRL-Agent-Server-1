FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    NPM_CONFIG_UPDATE_NOTIFIER=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@latest \
    && claude --version

# Claude Code refuses --dangerously-skip-permissions as root
RUN useradd -m -s /bin/bash polar && \
    mkdir -p /polar/session && chown -R polar:polar /polar/session
USER polar

WORKDIR /polar/session
