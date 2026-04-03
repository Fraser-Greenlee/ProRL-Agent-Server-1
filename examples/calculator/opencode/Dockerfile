FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    OPENCODE_FAKE_VCS=git

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g opencode-ai@latest \
    && opencode --version

WORKDIR /polar/session
