# SWE-Gym Example

This example scales the same packaged workflow up to a curated 10-task SWE-Gym sample.

Use it when you want:

- task-specific runtime images
- patch-based evaluation on a clean replay runtime
- one batch manifest plus per-task request and response files

## Topology

The example uses the same local cluster shape as the calculator demo, driven by [topology.yaml](topology.yaml).

Start it with:

```bash
uv run polar serve_rollout -c examples/swegym/topology.yaml
uv run polar serve_gateway -c examples/swegym/topology.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/swegym/topology.yaml --node-id localhost-node-02
```

## Host Setup

Install the host-side evaluator dependency once:

```bash
bash examples/swegym/setup_host.sh
```

## Build Images

Build the SWE-Agent images:

```bash
bash examples/swegym/swe_agent/setup.sh
```

Build the OpenHands SDK images:

```bash
bash examples/swegym/openhands_sdk/setup.sh
```

## Submit A Sample

Run one sampled task with one rollout:

```bash
uv run python examples/swegym/swe_agent/submit_tasks.py \
  --instance-id getmoto__moto-7365 \
  --max-tasks 1 \
  --num-rollouts 1
```

The helper writes task files, submits each one through `polar submit`, and stores a batch summary when the run completes.

## Outputs

```text
examples/swegym/<harness>/batches/<timestamp>/
  manifest.json
  summary.json
  <instance-id>/
    request.json
    response.json
```

## Notes

- The sample is text-only SWE-Gym data.
- `swe_agent` uses a dedicated `polar-sweagent` environment inside the derived image.
- `openhands_sdk` only builds on benchmark images whose native Python is already compatible.
