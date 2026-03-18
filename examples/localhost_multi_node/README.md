# Multi Node Localhost Rollout

This example runs one rollout server with a four-node gateway pool on the same
machine:

- two `vLLM` servers on `127.0.0.1:8000` and `127.0.0.1:8001`
- four `GatewayNode` processes on `127.0.0.1:8100` through `127.0.0.1:8103`
- one `Rollout Server` on `127.0.0.1:8080`

In this example, the four gateways are split evenly across the two `vLLM`
servers:

- `localhost-node-01` at `http://127.0.0.1:8100` -> `http://127.0.0.1:8000`
- `localhost-node-02` at `http://127.0.0.1:8101` -> `http://127.0.0.1:8000`
- `localhost-node-03` at `http://127.0.0.1:8102` -> `http://127.0.0.1:8001`
- `localhost-node-04` at `http://127.0.0.1:8103` -> `http://127.0.0.1:8001`

Each gateway advertises `capacity=4`, so the cluster capacity is `16` total
sessions.


## Prerequisites

- `claude` CLI installed and authenticated
- `vllm` installed
- access to `MiniMaxAI/MiniMax-M2.5`
- project virtualenv created:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Launch everything from the repo root.


## 1. Start Two vLLM Servers

Start two independent `vLLM` processes. Adjust GPU placement and tensor
parallel settings for your machine; the important part for this example is that
they listen on ports `8000` and `8001`.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --port 8000
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think  \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --port 8000
```

## 2. Start the Rollout Server

Terminal 3:

```bash
CONFIG_PATH=examples/localhost_multi_node/rollout_server.yaml \
.venv/bin/python -m rollout.server
```

## 3. Start the Gateway Pool

Terminal 4:

```bash
CONFIG_PATH=examples/localhost_multi_node/gateway_server.yaml \
.venv/bin/python -m gateway.server
```


## 4. Submit the Rollout Task

This request currently sends `32` rollouts into a cluster with total capacity
`16`, so the balancer will reuse nodes across multiple waves.

```bash
bash examples/localhost_multi_node/submit_task.sh
```

The request body is stored in `examples/localhost_multi_node/task_request.json`.



## Quick Node Checks

Cluster node status:

```bash
curl -sf http://127.0.0.1:8080/nodes | python3 -m json.tool
```

Gateway health:

```bash
curl -sf http://127.0.0.1:8100/health | python3 -m json.tool
curl -sf http://127.0.0.1:8101/health | python3 -m json.tool
curl -sf http://127.0.0.1:8102/health | python3 -m json.tool
curl -sf http://127.0.0.1:8103/health | python3 -m json.tool
```

Task status:

```bash
curl -sf http://127.0.0.1:8080/rollout/task/multi-node-localhost-capacity8 | python3 -m json.tool
```

One persisted result file:

```bash
python3 -m json.tool "$(ls rollout_results/task_multi-node-localhost-capacity8/rollout_*.json | head -n 1)"
```

## Expected Behavior

- all four gateways appear in `GET /nodes`
- sessions are dispatched across the least-loaded gateway nodes
- gateways `8100` and `8101` proxy to `vLLM :8000`
- gateways `8102` and `8103` proxy to `vLLM :8001`
- because this config sets `save_dir`, rollout results are also written under
  `./rollout_results/task_multi-node-localhost-capacity8/`
