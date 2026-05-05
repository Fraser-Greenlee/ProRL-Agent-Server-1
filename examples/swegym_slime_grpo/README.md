# SWE-Gym Slime GRPO

This example connects **Polar** rollout sessions with **Slime** GRPO training on
SWE-Gym tasks.

The exact model, topology, dependency pins, training arguments, worker counts,
ports, and harness settings are intentionally kept in the executable scripts and
YAML files. Treat those files as the source of truth, since the configuration is
expected to change as the example evolves.

## Quick Start

```bash
bash examples/swegym_slime_grpo/launch_e2e.sh
```

## Files

| File | Purpose |
|---|---|
| `launch_e2e.sh` | Single-entry launcher for setup and execution |
| `run.sh` | Main training and rollout launch script |
| `polar_config.yaml` | Polar rollout task template and harness configuration |
| `topology.yaml` | Polar cluster and gateway topology |
| `prepare_data.py` | Builds the SWE-Gym JSONL data used by Slime |
| `prepare_apptainer_images.py` | Prepares local Apptainer SIF images and shared agent CLI assets |
| `convert_weights.sh` | Converts model weights into the format used by Slime |
