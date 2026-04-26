ARG BASE_IMAGE=python:3.12-slim
ARG POLAR_SWEGYM_IMAGE_VERSION=4

# ── Stage 1: shared CLI layer ─────────────────────────────────────────
# Downloaded + npm-installed once on a tiny base; COPYed into all
# derived per-instance images. BuildKit / classic-cache keeps the stage
# reusable across builds so we don't repeat node+npm work for every task.
FROM debian:12-slim AS cli-layer
ENV PATH=/opt/node/bin:${PATH}
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/node \
    && curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz \
        | tar -xJ --strip-components=1 -C /opt/node \
    && node --version \
    && npm install -g --no-audit --no-fund --prefix=/opt/node \
        @openai/codex@latest \
        @anthropic-ai/claude-code@latest \
        @qwen-code/qwen-code@latest \
        opencode-ai@latest \
        @mariozechner/pi-coding-agent@latest \
    && ls /opt/node/bin

# ── Stage 2: per-instance runtime image ───────────────────────────────
FROM ${BASE_IMAGE}
ARG POLAR_SWEGYM_IMAGE_VERSION=4
LABEL io.polar.swegym-image-version=${POLAR_SWEGYM_IMAGE_VERSION}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/miniconda3/envs/testbed/bin:/opt/node/bin:${PATH}

COPY --from=cli-layer /opt/node /opt/node

RUN ln -sf /opt/node/bin/node /usr/local/bin/node \
    && ln -sf /opt/node/bin/npm /usr/local/bin/npm \
    && ln -sf /opt/node/bin/npx /usr/local/bin/npx \
    && for cli in codex claude qwen opencode pi; do \
        if [ -e /opt/node/bin/$cli ]; then \
            ln -sf /opt/node/bin/$cli /usr/local/bin/$cli; \
        fi; \
    done \
    && node --version \
    && codex --version 2>/dev/null || true \
    && claude --version 2>/dev/null || true \
    && qwen --version 2>/dev/null || true \
    && opencode --version 2>/dev/null || true \
    && pi --version 2>/dev/null || true

WORKDIR /testbed
