# SWE-Gym Example

End-to-end ARP example for a curated 10-task sample from
`NovaSky-AI/SkyRL-v0-293-data`.

This example ports the **per-instance SWE-Gym runtime image** and **SWE-style
patch evaluation** pattern into ARP's current architecture:

- each sampled task uses its own benchmark image derived from `instance_id`
- the agent runs inside that benchmark image
- evaluation uses ARP's `swegym_git_diff` evaluator with the task `instance`
- `refresh_runtime=true` replays the same runtime image and prepare flow before grading

Current scope:

- harnesses:
  - ARP built-in `swe_agent` CLI harness
  - ARP built-in `openhands_sdk` harness on the compatible benchmark-image subset
- evaluator: ARP built-in `swegym_git_diff`
- sample: 10 curated text-only SWE-Gym tasks across multiple repos

This is **not** a full ProRL/OpenHands-controller parity port yet. The
`openhands_sdk` variant is the closest currently practical OpenHands-flavored
setup under ARP's existing harness contract, while keeping the task-specific
benchmark image and evaluation semantics.

## Curated Sample

The curated sample uses these 10 `instance_id`s from the train split:

| Instance | Repo | Summary |
|----------|------|---------|
| `getmoto__moto-7365` | `getmoto/moto` | Fix Decimal arithmetic in DynamoDB mock updates |
| `python__mypy-10392` | `python/mypy` | Search `@python2` subdirectories in Python 2 mode |
| `conan-io__conan-13721` | `conan-io/conan` | Add `profile_name` to profile rendering |
| `iterative__dvc-1809` | `iterative/dvc` | Infer metrics type from file suffix |
| `dask__dask-10441` | `dask/dask` | Fix invalid append-mode handling in `to_csv` |
| `pydantic__pydantic-8072` | `pydantic/pydantic` | Remove `__pydantic_self__` constructor edge case |
| `pandas-dev__pandas-58335` | `pandas-dev/pandas` | Avoid incorrect `to_dict(orient="tight")` warning |
| `facebookresearch__hydra-1783` | `facebookresearch/hydra` | Improve missing-default error messaging |
| `bokeh__bokeh-13636` | `bokeh/bokeh` | Use globally unique, CSS-safe embedded JSON IDs |
| `Project-MONAI__MONAI-2238` | `Project-MONAI/MONAI` | Prevent `CopyItemsd` from reusing the same key |

## Topology

This example uses the same localhost topology as the calculator demo:

| Component | Address |
|-----------|---------|
| Rollout server | `http://127.0.0.1:8080` |
| Gateway node 01 | `http://127.0.0.1:8100` → vLLM `:8000` |
| Gateway node 02 | `http://127.0.0.1:8101` → vLLM `:8001` |

Both vLLM servers should host `MiniMaxAI/MiniMax-M2.5`.

## Prerequisites

1. **Two vLLM servers** on ports `8000` and `8001`.
2. **Docker** available for pulling the benchmark images and building the derived
   harness images. For `apptainer` runtime sessions, the submitter rewrites local
   Docker image tags to `docker-daemon:<tag>` URIs.
3. **ARP installed** in a Python environment with access to `src/`.
4. **SWE-Gym evaluator package** installed in the same Python environment used to
   run the rollout and gateway servers.

## One-Time Host Setup

Activate your ARP environment first, then set `ARP_PYTHON` to that interpreter:

```bash
export ARP_PYTHON=/path/to/your/arp-env/bin/python
```

Install the host-side evaluator dependency:

```bash
bash examples/swegym/shared/setup_host.sh
```

This installs:

```bash
git+https://github.com/SWE-Gym/SWE-Bench-Package.git
```

## Starting The Servers

```bash
# Terminal 1 — rollout server
CONFIG_PATH=examples/swegym/shared/rollout_server.yaml \
  "$ARP_PYTHON" -m rollout.server

# Terminal 2 — gateway node 01
CONFIG_PATH=examples/swegym/shared/gateway_server.yaml \
  GATEWAY_NODE_ID=localhost-node-01 \
  "$ARP_PYTHON" -m gateway.server

# Terminal 3 — gateway node 02
CONFIG_PATH=examples/swegym/shared/gateway_server.yaml \
  GATEWAY_NODE_ID=localhost-node-02 \
  "$ARP_PYTHON" -m gateway.server
```

## Building The Derived Images

Each sampled task needs a derived runtime image layered on top of the
corresponding SWE-Gym benchmark image.

### `swe_agent`

Important detail for the `swe_agent` variant:

- the benchmark image's `testbed` env can be on older Python versions for some
  repos
- the derived image therefore creates a separate `arp-sweagent` conda env for
  the agent tooling
- the agent edits a copied repo at `/arp/session/workspace`
- evaluation still runs against `/testbed`

Build all 10 images:

```bash
bash examples/swegym/swe_agent/setup.sh
```

Build only one sampled task image:

```bash
bash examples/swegym/swe_agent/setup.sh \
  --instance-id getmoto__moto-7365
```

### `openhands_sdk`

Build the compatible `openhands_sdk` images:

```bash
bash examples/swegym/openhands_sdk/setup.sh
```

Important detail for the `openhands_sdk` variant:

- current `openhands-sdk` releases require Python `>=3.12`
- the setup script checks the benchmark image's native `testbed` Python
- incompatible tasks are skipped instead of failing the whole build

## Running The Example

Submit one rollout for one sampled task:

```bash
PYTHONPATH=src "$ARP_PYTHON" examples/swegym/swe_agent/submit_tasks.py \
  --instance-id getmoto__moto-7365 \
  --max-tasks 1 \
  --num-rollouts 1
```

Submit one rollout for all 10 sampled tasks:

```bash
PYTHONPATH=src "$ARP_PYTHON" examples/swegym/swe_agent/submit_tasks.py \
  --max-tasks 10 \
  --num-rollouts 1
```

Run multiple rollouts per task:

```bash
PYTHONPATH=src "$ARP_PYTHON" examples/swegym/swe_agent/submit_tasks.py \
  --max-tasks 10 \
  --num-rollouts 2
```

Run the `openhands_sdk` variant on one explicitly built task:

```bash
PYTHONPATH=src "$ARP_PYTHON" examples/swegym/openhands_sdk/submit_tasks.py \
  --instance-id getmoto__moto-7365 \
  --num-rollouts 1
```

Run the `openhands_sdk` variant on all locally built compatible tasks:

```bash
PYTHONPATH=src "$ARP_PYTHON" examples/swegym/openhands_sdk/submit_tasks.py \
  --max-tasks 10 \
  --num-rollouts 1
```

## What The Submitter Does

For each sampled instance, the submitter:

1. fetches the full task metadata from `NovaSky-AI/SkyRL-v0-293-data`
2. maps `instance_id` to the benchmark image
3. uses a harness-specific local image tag such as:
   - `arp-swegym-swe_agent:<instance-tag>`
   - `arp-swegym-openhands_sdk:<instance-tag>`
4. copies `/testbed` to `/arp/session/workspace` for the agent run
5. configures harness-specific runtime env:
   - `swe_agent`: dedicated `arp-sweagent` env on top of the benchmark image
   - `openhands_sdk`: dedicated `/opt/openhands-sdk-venv` built from the
     benchmark image's native `testbed` Python when compatible
6. evaluates with:
   - `strategy: swegym_git_diff`
   - `instance: <task-metadata>`
   - `repo_dir: /testbed`
   - `refresh_runtime: true`

Outputs are written to:

```text
examples/swegym/<harness>/batches/<timestamp>/
  manifest.json
  summary.json
  <instance-id>/
    request.json
    response.json
```

## Notes

- The NovaSky 293-task sample is text-only SWE-Gym data. It does **not** use
  `image_assets`, so these examples pull benchmark container images rather than
  visual assets.
- The build step can take a while because each selected task pulls a benchmark
  image and then layers agent tooling on top.
- If you only want a quick smoke test, start with `getmoto__moto-7365`.
- `openhands_sdk` currently depends on benchmark images whose native Python is
  already `>=3.12`. Older benchmark images are skipped by its setup script.

## Validation

This example was smoke-tested on:

- `instance_id`: `getmoto__moto-7365`
- harnesses:
  - `swe_agent`
  - `openhands_sdk`
- model: `MiniMaxAI/MiniMax-M2.5`
- topology: localhost rollout + two gateway nodes on `:8080`, `:8100`, `:8101`

Observed result:

- the rollout sessions completed successfully
- the gateway captured model traffic for the session
- the fresh evaluator runtime no longer collided with the agent runtime
- the sampled rollouts produced `reward=0.0` in smoke tests because the model
  explored but did not generate a fixing patch
