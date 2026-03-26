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

RUN npm install -g @qwen-code/qwen-code@latest \
    && qwen --version

WORKDIR /arp/session
