# CUA Data Collection

## How to Run (Interactive Session for Debugging)
### 1. Boot Up vLLM Servers
We use two vLLM servers - 1 for Qwen3-VL-235B (goal generation policy, aka `explorer policy` in the code) and 1 for
UI-TARS-1.5-7B (action generation policy, aka `uitars policy` in the code).

Run both servers on 2 nodes by
```bash
cd scripts
sbatch run_model.sbatch
```

The logs will be shown at `logs/explorer.out` and `logs/uitars.out`.

### 2. Run CPU node for data collection, with /dev/kvm write permission
We now run CPU interactive session, in which we boot up the linux virtual machine (VM) and call the vLLM servers to
collect the trajectories. Here, it's important we have a write access to `/dev/kvm`, as it allows us to accelerate VM.

Run the bash script to spin up the CPU interactive node by
```bash
cd scripts
bash debug_interactive.sh
```

***What does `debug_interactive.sh` do?*** \
(1) We first allocate 1 CPU interactive node (with `sleep infinity &` as the command). We assign one of the reserved nodes
for `/dev/kvm` access. \
(2) When the node is ready with enroot container running, we ssh into the node, and fetch the enroot container ID. \
(3) We run `enroot exec $CONTAINER_ID bash` in order to access a bash shell inside that enroot container.

***Wait, why aren't we just directly using the CPU interactive node in step (1)?*** \
This is a very finicky detail, but in step (1), the enroot container environment loses `/dev/kvm` access that the
CPU node originally had. The only way to retain `/dev/kvm` access inside enroot is to first boot up the container,
then running `enroot exec $CONTAINER_ID bash` from outside.


### 3. Run Data Collection Script for Debugging
```bash
python debug_collect_trajectories.py --uitars_node $UITARS_NODE --explorer_node $EXPLORER_NODE
```
`$UITARS_NODE` and `$EXPLORER_NODE` should be manually set by the user (e.g., pool0-2838).

## To-Do

- **Testing UI-TARS** \
UI-TARS often generates tool calls slightly different from the instruction it has been given. We need to test if there
is any missing case in our inference logic, implemented in `module_uitars_controller.py`.


- **Implementing parallelism with `asyncio` and `threadpoolexecutor`** \
The current implementation is sequential for debugging. We need to implement a parallel version, in a similar fashion to
our old pipeline in `synthetic_data_generator.py`.


