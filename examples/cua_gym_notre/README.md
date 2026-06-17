# CUA-Gym Web (Notre / Hermes)

Rollout-only example: run the shared [`notre_hermes`](../../src/polar/agent/presets/notre_hermes.py)
agent over the web portion of [CUA-Gym](https://huggingface.co/datasets/xlangai/CUA-Gym)
and grade the resulting app state with the
[`env_state`](../../src/polar/trajectory/evaluator/env_state.py) evaluator.

- **Dataset:** `xlangai/CUA-Gym` (web / `mock_web` / `setup_kind=py` tasks).
- **Apps:** [CUA-Gym-Hub](https://github.com/xlang-ai/CUA-Gym-Hub) — Vite mock web
  apps (Slack, Amazon, …), cloned at prepare time and served per task.
- **Environment:** [CUA-Gym](https://github.com/xlang-ai/CUA-Gym). Each task's
  `initial_setup.py` seeds the app state and `reward.py` scores the final state
  via the app's `GET /go?sid=` endpoint (prints `REWARD: <float>`).
- **Agent:** `notre_hermes` with the `[web]` extras — drives the live UI through
  Hermes' browser toolset (agent-browser + Chromium).

> Note: tasks whose `reward.py` calls the CUA LLM-as-judge helper need that
> helper deployed; prepared tasks are filtered to single-app web tasks and are
> mostly programmatically graded.

## 1. Build the runtime image

```bash
uv run python examples/cua_gym_notre/build_image.py
```

## 2. Prepare data (clone apps + materialize bundles)

```bash
uv run python examples/cua_gym_notre/prepare_data.py --max-tasks 10
uv run python examples/cua_gym_notre/prepare_data.py --max-tasks 50 --difficulty easy,medium
```

This clones CUA-Gym-Hub into `data/cua_gym_hub/`, downloads the task archive,
assigns each task a local Vite port, and writes `data/cua_web_tasks.jsonl`.
Requires `git` and the `zstd` CLI on the host.

## 3. Serve the model and start Polar

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3.6-27B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3

polar serve_rollout -c examples/cua_gym_notre/topology.vllm.yaml
polar serve_gateway -c examples/cua_gym_notre/topology.vllm.yaml --node-id localhost-node-01
polar serve_gateway -c examples/cua_gym_notre/topology.vllm.yaml --node-id localhost-node-02
```

## 4. Submit rollouts

```bash
uv run python examples/cua_gym_notre/submit_cua_tasks.py --max-tasks 10
uv run python examples/cua_gym_notre/submit_cua_tasks.py --max-tasks 50 --num-samples 4
```

Each task serves its app on a dedicated port (host network), seeds state, lets
the agent act in the browser, then `reward.py` scores the final state. Watch live
progress with `polar dashboard -c examples/cua_gym_notre/topology.vllm.yaml`.
