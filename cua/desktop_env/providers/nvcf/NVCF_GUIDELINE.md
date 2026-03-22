# NVCF Provider Configuration Guide

This guide explains how to configure and run OSWorld with the `nvcf` provider.

The `nvcf` provider supports:

- **Attach mode**: connect to an existing NVCF function (`NVCF_FUNCTION_ID`).
- **Auto-deploy mode**: create + deploy a function automatically when starting.
- **Auto-undeploy**: undeploy the function on shutdown if this session deployed it.

---

## Overview

The provider runs a local authenticated proxy to NVIDIA Cloud Functions (NVCF):

- `http://127.0.0.1:<PORT>/api/*` -> `https://grpc.nvcf.nvidia.com/api/*`
- `http://127.0.0.1:<PORT>/chrome/*` -> `https://grpc.nvcf.nvidia.com/chrome/*`
- `http://127.0.0.1:<PORT>/vlc/*` -> `https://grpc.nvcf.nvidia.com/vlc/*`

This matches the same single-port path-prefix style used by `nvcf_dummy`.

---

## Prerequisites

1. **NGC API key** with permission to invoke/deploy cloud functions.
2. **Python dependencies**:
   - `aiohttp` (local HTTP+WebSocket proxy)
   - `ngcsdk` (required only for auto-deploy)
3. If using auto-deploy, you need your **NGC org name**.

---

## Environment Variables

### Required (all modes)

```bash
export NGC_API_KEY="nvapi-xxxx"
export NGC_ORG="your-ngc-org"
```

### Attach mode (reuse existing function)

```bash
export NVCF_FUNCTION_ID="your-function-id"
# Optional:
export NVCF_VERSION_ID="your-version-id"
```

When `NVCF_FUNCTION_ID` is set, the provider uses that function directly and does not create a new one.

### Auto-deploy mode (create/deploy automatically)

Optional tuning:

```bash
# Defaults to: nvcr.io/${NGC_ORG}/nemo:osworld-linux-eval
export NVCF_CONTAINER_IMAGE="nvcr.io/${NGC_ORG}/nemo:osworld-linux-eval"
export NVCF_GPU="L40S"
export NVCF_INSTANCE_TYPE=""                  # optional explicit instance type
export NVCF_MIN_INSTANCES="1"
export NVCF_MAX_INSTANCES="1"
export NVCF_MAX_REQUEST_CONCURRENCY="1"
export NVCF_BACKEND="GFN"

export NVCF_DEPLOY_TIMEOUT="1800"             # seconds
export NVCF_DEPLOY_POLL_INTERVAL="30"         # seconds
```

### Proxy + teardown behavior

```bash
export NVCF_LOCAL_PROXY_PORT="8090"           # base port; provider picks next free if occupied
export NVCF_READY_TIMEOUT="300"               # seconds

export NVCF_UNDEPLOY_ON_STOP="1"              # default true
export NVCF_DELETE_FUNCTION_ON_STOP="0"       # default false
```

---

## Running with NVCF

### Human mode

```bash
python main.py --provider nvcf
```

### Evaluation mode

```bash
python run.py --provider_name nvcf --path_to_vm ""
```

`path_to_vm` is ignored for this provider.

---

## Behavior Notes

- The provider returns a synthetic endpoint tuple: `localhost:<PORT>:<PORT>:8006:<PORT>`.
- `DesktopEnv` maps it to:
  - API server: `<PORT>/api`
  - Chrome CDP: `<PORT>/chrome`
  - VLC: `<PORT>/vlc`
- VNC is not proxied through NVCF in this provider; `8006` is a compatibility placeholder.

---

## Troubleshooting

### `NGC_API_KEY is required`

Set `NGC_API_KEY` in your shell before starting.

### Auto-deploy fails with org error

Set `NGC_ORG` in your environment. The `nvcf` provider requires it in all modes.

### `ngcsdk is required for NVCF auto-deploy`

Install dependencies and retry:

```bash
pip install ngcsdk aiohttp
```

### Proxy port already in use

Set a different base port:

```bash
export NVCF_LOCAL_PROXY_PORT="18090"
```

### Deploy timeout

Increase deployment timeout and poll interval:

```bash
export NVCF_DEPLOY_TIMEOUT="3600"
export NVCF_DEPLOY_POLL_INTERVAL="30"
```

### Chrome/CDP connection issues

The provider rewrites Chrome discovery `ws://localhost/...` URLs to local proxy URLs. If issues persist, verify:

- function is ACTIVE,
- `/chrome/json/version` is reachable via local proxy,
- no local firewall/process is blocking the proxy port.

---

## Cost and Cleanup

- Auto-deploy can incur cloud charges while instances are running.
- By default, provider undeploys functions it created on shutdown.
- Set `NVCF_DELETE_FUNCTION_ON_STOP=1` if you want to remove the function version as well.

