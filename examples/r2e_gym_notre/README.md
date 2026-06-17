# R2E-Gym-Subset (Notre / Hermes)

Rollout-only example: run the shared [`notre_hermes`](../../src/polar/agent/presets/notre_hermes.py)
agent over the full [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset)
coding dataset and grade each patch with the [`test_harness`](../../src/polar/trajectory/evaluator/test_harness.py)
evaluator.

- **Dataset:** `R2E-Gym/R2E-Gym-Subset` — each instance carries a ready-to-run
  Docker image with the repo at the buggy commit under `/testbed` and the gold
  suite under `/r2e_tests`.
- **Environment:** [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym). The image *is*
  the runtime; the agent edits `/testbed`, then the evaluator re-runs
  `python -m pytest r2e_tests` against the patch in a fresh container.
- **Agent:** `notre_hermes` (Hermes + `terminal,file,code_execution,browser,web,search`
  toolset). Hermes is pip-installed into an isolated venv during INIT.

## 1. Cache the dataset and pull images

```bash
uv run python examples/r2e_gym_notre/pull_images.py --max-tasks 10
```

R2E-Gym images are large; pull only what you plan to run (`--max-tasks`, or
repeat `--instance-id`). Omit `--max-tasks` to pull the whole subset.

## 2. Serve the model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3.6-27B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 8 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Run a second instance on port `8001` to feed both gateway nodes, or trim
`topology.vllm.yaml` to one node. Use `topology.sgl.yaml` for an SGLang backend.

## 3. Start Polar

```bash
polar serve_rollout -c examples/r2e_gym_notre/topology.vllm.yaml
polar serve_gateway -c examples/r2e_gym_notre/topology.vllm.yaml --node-id localhost-node-01
polar serve_gateway -c examples/r2e_gym_notre/topology.vllm.yaml --node-id localhost-node-02
```

## 4. Submit rollouts

```bash
uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --max-tasks 10
uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --max-tasks 50 --num-samples 4
uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --instance-id <instance_id>
```

Tasks that resolve (gold tests pass on the patched repo) score `1.0`. Watch live
progress with `polar dashboard -c examples/r2e_gym_notre/topology.vllm.yaml`.
