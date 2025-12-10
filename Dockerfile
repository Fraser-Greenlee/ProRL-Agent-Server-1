# Start from the NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.11-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Define installation arguments
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

# Install systemctl
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd netcat && \
    apt-get clean

# Install tini
RUN apt-get update && \
    apt-get install -y tini && \
    apt-get clean

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip


WORKDIR /workspace

# install enroot
# Install NVIDIA Enroot
# RUN arch=$(dpkg --print-architecture) && \
#     curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot_3.5.0-1_${arch}.deb && \
#     curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot+caps_3.5.0-1_${arch}.deb && \
#     apt install -y ./*.deb && \
#     rm -f *.deb
# # Disable NVIDIA hook in Enroot
# RUN mv /etc/enroot/hooks.d/98-nvidia.sh /etc/enroot/hooks.d/98-nvidia.sh.disabled

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url

RUN pip3 install pynvml immutabledict nltk langdetect absl-py
RUN python3 -c "import nltk; nltk.download('punkt_tab')"
RUN apt-get update && \
    apt-get install -y gosu sudo tmux singularity-container fuse && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true
RUN sudo su -
RUN chmod 777 -R /root
RUN pip install math-verify[antlr4_9_3] torchdata pudb


##################### OpenHands #######################
# Install OpenHands dependencies from pyproject.toml
RUN pip install --no-cache-dir \
    "litellm>=1.60.0,!=1.64.4,!=1.67.*" \
    "aiohttp>=3.9.0,!=3.11.13" \
    google-generativeai \
    "google-api-python-client>=2.164.0" \
    google-auth-httplib2 \
    google-auth-oauthlib \
    termcolor \
    docker \
    fastapi \
    toml \
    uvicorn \
    types-toml \
    pydrive   \
    formulas \
    lxml \
    cssselect \
    xmltodict \
    tldextract \
    pymupdf \
    borb \
    imagehash \
    easyocr \
    odfpy \
    pdfplumber \
    mutagen \
    pyacoustid \
    numpy==1.26.4 \
    "transformers>=4.46,<4.50" \
    "beartype>=0.20.0" \
    daytona_api_client==0.20.1 \
    json-repair \
    "browsergym-core==0.13.3" \
    html2text \
    "e2b>=1.0.5,<1.4.0" \
    pexpect \
    "jinja2>=3.1.3" \
    python-multipart \
    boto3 \
    "minio>=7.2.8" \
    "tenacity>=8.5,<10.0" \
    "zope-interface==7.2" \
    "pathspec>=0.12.1" \
    google-cloud-aiplatform \
    "anthropic[vertex]" \
    "tree-sitter>=0.24.0" \
    "bashlex>=0.18" \
    "pyjwt>=2.9.0" \
    dirhash \
    "python-frontmatter>=1.1.0" \
    python-docx \
    PyPDF2 \
    python-pptx \
    pylatexenc \
    "tornado<=6.2" \
    python-dotenv \
    "rapidfuzz>=3.9.0" \
    "whatthepatch>=1.0.6" \
    "protobuf>=4.21.6,<5.0.0" \
    "opentelemetry-api==1.25.0" \
    "opentelemetry-exporter-otlp-proto-grpc==1.25.0" \
    "modal>=0.66.26,<0.78.0" \
    "runloop-api-client==0.33.0" \
    "libtmux>=0.37,<0.40" \
    "pygithub>=2.5.0" \
    joblib \
    "openhands-aci==0.3.0" \
    "python-socketio>=5.11.4" \
    "redis>=5.2,<7.0" \
    "sse-starlette>=2.1.3" \
    psutil \
    "stripe>=11.5,<13.0" \
    "ipywidgets>=8.1.5" \
    "qtconsole>=5.6.1" \
    "memory-profiler>=0.61.0" \
    "daytona-sdk==0.18.1" \
    "python-json-logger>=3.2.1" \
    "prompt-toolkit>=3.0.50" \
    "poetry>=2.1.2" \
    "anyio==4.9.0" \
    pythonnet \
    "fastmcp>=2.5.2" \
    "mcpm==1.12.0"

# Install runtime dependencies
# NOTE: jupyter_kernel_gateway removed - requires tornado>=6.4 which conflicts with tornado<=6.2
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    flake8

# Install evaluation dependencies (optional - comment out if not needed)
RUN pip install --no-cache-dir \
    streamlit \
    whatthepatch \
    retry \
    evaluate \
    "swebench>=3.0.8" \
    commit0 \
    func_timeout \
    sympy \
    gdown \
    matplotlib \
    seaborn \
    tabulate \
    "browsergym-miniwob==0.13.3" \
    "boto3-stubs[s3]>=1.37.19" \
    "pyarrow>=14,<18" \
    datasets

# Install test dependencies
RUN pip install --no-cache-dir \
    "pytest-asyncio" \
    "pytest-cov" \
    "pytest-forked" \
    "pytest-xdist" \
    "openai" \
    "reportlab" \
    "gevent>=24.2.1,<26.0.0"

RUN playwright install-deps
RUN playwright install

RUN pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git 
# 2. Install add-apt-repository prerequisite
RUN apt-get install -y software-properties-common

# 3. Add the Apptainer official PPA
RUN add-apt-repository -y ppa:apptainer/ppa

# 4. Update and install Apptainer
RUN apt-get update && \
    apt-get install -y apptainer

# Add NodeSource repository
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -

# Install Node.js and npm
RUN sudo apt-get install -y nodejs

# install python requirements from host machine pip install -r ./osworld/desktop_env/server/requirements.txt
COPY ./openhands/nvidia/os_world/requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
