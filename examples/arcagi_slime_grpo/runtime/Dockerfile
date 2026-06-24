# ARC-AGI compression runtime for Polar rollouts.
#
# Bakes:
#   - the auto-compress repo at /opt/auto-compress (eval.py, library.hy,
#     tasks/arcagi/train/*.hy scaffolds, reference/)
#   - /opt/arcagi-venv: hy + hyrule + arckit + scipy + numpy (what eval.py needs)
#   - /opt/openhands-sdk-venv: openhands-sdk/tools 1.17 (the harness fallback
#     interpreter — baked so we don't pip-install on every rollout)
#   - node 22 (OpenHands SDK + any MCP servers)
#
# Build context is the auto-compress repo root (see build_image.py).
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ARG POLAR_ARCAGI_IMAGE_VERSION=1

COPY --from=node /usr/local/ /usr/local/

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.local \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    ARCAGI_VENV=/opt/arcagi-venv \
    AUTO_COMPRESS=/opt/auto-compress \
    PATH=/home/polar/.local/bin:/opt/arcagi-venv/bin:${PATH}

LABEL io.polar.arcagi-image-version="${POLAR_ARCAGI_IMAGE_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        sudo \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash polar \
    && echo 'polar ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /polar/session /home/polar/.local \
    && chown -R polar:polar /polar/session /home/polar

# ── arcagi eval venv ─────────────────────────────────────────────────
# Only the runtime deps eval.py / library.hy import — no drawsvg/rich (those
# are for local --show rendering, not needed in the rollout grader path).
RUN python3 -m venv /opt/arcagi-venv \
    && /opt/arcagi-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/arcagi-venv/bin/pip install --no-cache-dir \
        "hy>=1.0.0" "hyrule>=0.7" "numpy>=2.0" "scipy>=1.13" "arckit>=1.0.1"

# ── OpenHands SDK venv (harness fallback interpreter) ────────────────
# The openhands_sdk preset looks for $HOME/.venv/bin/python, then
# /opt/openhands-sdk-venv/bin/python.  Bake the latter so rollouts don't
# pip-install on every session.  Pin sdk+tools to 1.17.0 (1.18+ needs 3.13).
RUN python3 -m venv /opt/openhands-sdk-venv \
    && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir \
        openhands-sdk==1.17.0 openhands-tools==1.17.0

# ── auto-compress repo ───────────────────────────────────────────────
# Copy the contract (eval.py + library.hy), the task scaffolds, and the
# reference material the prompt cites.  .dockerignore trims the heavy/irrelevant
# bits (.git, .venv, runs/, sft/, results).
COPY --chown=polar:polar . /opt/auto-compress

# Warm arckit's dataset cache so rollouts don't hit the network. arckit
# downloads ARC json on first load; do it once at build time.
RUN /opt/arcagi-venv/bin/python -c "import arckit; arckit.load_data()" || true

# Reset EVERY task file to its baseline scaffold.  The source repo ships
# already-compressed solutions; baking those would (a) leak answers to the
# agent via sibling `tasks/arcagi/train/*.hy` it's told it may read, and
# (b) defeat the point of compressing from scratch.  `--scaffold` rewrites
# all task files in place with the raw-grid baseline. The per-rollout
# `prepare` step still re-writes the target task's baseline_body, so this
# guarantees the whole tasks/ tree is solution-free at training start.
RUN cd /opt/auto-compress \
    && /opt/arcagi-venv/bin/python eval.py --scaffold

# For the synthetic rl_tasks image: reset every sft/rl_tasks/<id>/<id>.hy to its
# raw-grid baseline (so the agent compresses from scratch and no reference
# solution leaks). pairs.json (ground truth) is kept — the grader reads it but
# the diff filter blocks the agent from editing it. Gated so the arcagi image
# is unaffected (and sft/ may be absent there per .dockerignore).
ARG SCAFFOLD_RL_TASKS=0
RUN if [ "$SCAFFOLD_RL_TASKS" = "1" ] && [ -d /opt/auto-compress/sft/rl_tasks ]; then \
        cd /opt/auto-compress && \
        for d in sft/rl_tasks/*/; do \
            id="$(basename "$d")"; \
            [ -f "$d/pairs.json" ] || continue; \
            /opt/arcagi-venv/bin/python sft/rl_tasks/score_synthetic.py --scaffold "$id" >/dev/null || true; \
        done && \
        echo "scaffolded synthetic rl_tasks to baseline"; \
    fi

# Persist PATH for login shells (docker exec bash -lc).
RUN echo 'export PATH=/home/polar/.local/bin:/opt/arcagi-venv/bin:$PATH' \
    > /etc/profile.d/polar-path.sh

USER polar
WORKDIR /polar/session
