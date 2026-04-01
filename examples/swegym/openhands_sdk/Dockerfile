ARG BASE_IMAGE=python:3.12-slim
ARG ARP_SWEGYM_IMAGE_VERSION=1

FROM ${BASE_IMAGE}
ARG ARP_SWEGYM_IMAGE_VERSION

LABEL io.nv-arp.swegym-image-version="${ARP_SWEGYM_IMAGE_VERSION}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN /opt/miniconda3/envs/testbed/bin/python - <<'PY'
import sys
assert sys.version_info >= (3, 12), sys.version
PY

RUN /opt/miniconda3/envs/testbed/bin/python -m venv /opt/openhands-sdk-venv \
    && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir openhands-sdk openhands-tools

WORKDIR /arp/session
