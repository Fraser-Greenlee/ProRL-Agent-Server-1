# Start from the NVIDIA official image
FROM nvcr.io/nvidia/pytorch:24.11-py3

# -----------------------------------------------------------------------------
# 1. Setup Environment & UV
# -----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV MAX_JOBS=32 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DEBIAN_FRONTEND=noninteractive \
    NODE_OPTIONS="" \
    HF_HUB_ENABLE_HF_TRANSFER="1" \
    # UV Configuration
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Python Configuration
    PYTHONUNBUFFERED=1

# Define build arguments for Mirrors
ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Configure UV to use the requested mirror
ENV UV_INDEX_URL=${PIP_INDEX}


# -----------------------------------------------------------------------------
# 2. System Dependencies (Grouped)
# -----------------------------------------------------------------------------
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} noble main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} noble-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} noble-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} noble-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:apptainer/ppa
# 4. Update and install Apptainer
RUN apt-get update && \
    apt-get install -y apptainer

RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-confdef" \
    systemd \
    tini \
    gosu \
    sudo \
    tmux \
    fuse \
    git \
    curl \
    software-properties-common \
    # Required for PDF/Image processing (easyocr, pymupdf, etc)
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

# Install Node.js 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Install NVIDIA Enroot
RUN arch=$(dpkg --print-architecture) && \
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot_3.5.0-1_${arch}.deb && \
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.5.0/enroot+caps_3.5.0-1_${arch}.deb && \
    apt install -y ./*.deb && \
    rm -f *.deb && \
    mv /etc/enroot/hooks.d/98-nvidia.sh /etc/enroot/hooks.d/98-nvidia.sh.disabled



RUN rm -rf /usr/lib/python3.*/EXTERNALLY-MANAGED
# -----------------------------------------------------------------------------
# 3. Python Dependency Cleaning & Setup
# -----------------------------------------------------------------------------
# Uninstall conflicts from the base NVIDIA image
RUN uv pip uninstall \
    torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core \
    pynvml nvidia-ml-py

# -----------------------------------------------------------------------------
# 4. Install Torch Stack (vLLM + FlashAttn)
# -----------------------------------------------------------------------------
# Install Base Core (Torch + vLLM)
RUN uv pip install --no-cache \
    "vllm==0.8.5" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "tensordict==0.6.2" \
    "numpy<2.0.0" \
    torchdata \
    "transformers[hf_xet]>=4.51.0" \
    accelerate datasets peft hf-transfer \
    "pyarrow>=15.0.0" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff \
    # Fix packages
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    # NLP Utils
    immutabledict nltk langdetect absl-py math-verify[antlr4_9_3] pudb

# Install Pre-built Wheels (Flash Attention & FlashInfer)
RUN uv pip install --no-cache \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu126torch2.6-cp38-abi3-linux_x86_64.whl

# -----------------------------------------------------------------------------
# 5. Install OpenHands & Application Stack
# -----------------------------------------------------------------------------
# Consolidated requirement list for better resolution
COPY <<EOF /tmp/openhands_reqs.txt
litellm>=1.60.0,!=1.64.4,!=1.67.*
aiohttp>=3.9.0,!=3.11.13
google-generativeai
google-api-python-client>=2.164.0
google-auth-httplib2
google-auth-oauthlib
termcolor
docker
fastapi
toml
uvicorn
types-toml
numpy<2.0.0
json-repair
browsergym-core==0.13.3
html2text
e2b>=1.0.5,<1.4.0
pexpect
jinja2>=3.1.3
python-multipart
boto3
minio>=7.2.8
tenacity>=8.5,<10.0
zope-interface==7.2
pathspec>=0.12.1
google-cloud-aiplatform
anthropic[vertex]
tree-sitter>=0.24.0
bashlex>=0.18
pyjwt>=2.9.0
dirhash
python-frontmatter>=1.1.0
python-docx
PyPDF2
python-pptx
pylatexenc
tornado
python-dotenv
rapidfuzz>=3.9.0
whatthepatch>=1.0.6
protobuf>=4.21.6,<5.0.0
opentelemetry-api==1.25.0
opentelemetry-exporter-otlp-proto-grpc==1.25.0
modal>=0.66.26,<0.78.0
runloop-api-client==0.33.0
libtmux>=0.37,<0.40
pygithub>=2.5.0
joblib
openhands-aci==0.3.0
python-socketio>=5.11.4
redis>=5.2,<7.0
sse-starlette>=2.1.3
psutil
stripe>=11.5,<13.0
ipywidgets>=8.1.5
qtconsole>=5.6.1
memory-profiler>=0.61.0
daytona-sdk==0.18.1
daytona_api_client==0.20.1
python-json-logger>=3.2.1
prompt-toolkit>=3.0.50
poetry>=2.1.2
anyio==4.9.0
pythonnet
fastmcp>=2.5.2
mcpm==1.12.0
jupyterlab
notebook
jupyter_kernel_gateway
flake8
streamlit
retry
evaluate
swebench>=3.0.8
commit0
func_timeout
sympy
gdown
matplotlib
seaborn
tabulate
browsergym==0.13.3
browsergym-webarena==0.13.3
browsergym-miniwob==0.13.3
browsergym-visualwebarena==0.13.3
boto3-stubs[s3]>=1.37.19
pyarrow==20.0.0
datasets
pytest-asyncio
pytest-cov
pytest-forked
pytest-xdist
openai
reportlab
gevent>=24.2.1,<26.0.0
# --- Newly Added Libraries ---
pydrive
formulas
lxml
cssselect
xmltodict
tldextract
pymupdf
borb
imagehash
easyocr
odfpy
pdfplumber
mutagen
pyacoustid
fastdtw
EOF

RUN uv pip install --no-cache -r /tmp/openhands_reqs.txt

# -----------------------------------------------------------------------------
# 6. Post-Installation & Git Installs
# -----------------------------------------------------------------------------

# Install SWE-Bench from Git
RUN uv pip install --no-cache git+https://github.com/SWE-Gym/SWE-Bench-Package.git

# NLTK Download
RUN python3 -c "import nltk; nltk.download('punkt_tab')"

# Playwright
RUN playwright install-deps && playwright install

# Final Setup
WORKDIR /workspace
RUN chmod 777 -R /root
