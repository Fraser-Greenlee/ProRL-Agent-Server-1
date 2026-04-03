ARG BASE_IMAGE=python:3.12-slim
ARG POLAR_SWEGYM_IMAGE_VERSION=2
FROM ${BASE_IMAGE}

ARG POLAR_SWEGYM_IMAGE_VERSION=2
LABEL io.polar.swegym-image-version=${POLAR_SWEGYM_IMAGE_VERSION}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/envs/polar-sweagent/bin:${PATH}

RUN /opt/miniconda3/bin/conda create -y -n polar-sweagent python=3.11 pip \
    && /opt/miniconda3/envs/polar-sweagent/bin/python -m pip install --no-cache-dir "git+https://github.com/SWE-agent/SWE-agent.git" \
    && SITE=$(/opt/miniconda3/envs/polar-sweagent/bin/python -c "import site; print(site.getsitepackages()[0])") \
    && git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git /tmp/swe-agent-src \
    && cp -r /tmp/swe-agent-src/config "$SITE/config" \
    && cp -r /tmp/swe-agent-src/tools "$SITE/tools" \
    && mkdir -p "$SITE/trajectories" \
    && /opt/miniconda3/bin/conda clean -afy \
    && rm -rf /tmp/swe-agent-src

WORKDIR /testbed
