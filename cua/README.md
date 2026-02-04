# CUA Data Collection

## Running on Interactive Session
### 1. Boot Up vLLM Servers
We use two vLLM servers - 1 for Qwen3-VL-235B (goal generation policy, aka `planner model` in the code) and 1 for
UI-TARS-1.5-7B (action generation policy, aka `actor model` in the code).

Run both servers on 2 nodes by
```bash
cd scripts
sbatch run_model.sbatch
```

The logs will be shown at `scripts/logs/planner.out` and `scripts/logs/actor.out`.

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


### 3-A. Run Data Collection Script for Debugging
```bash
python debug_collect_trajectories.py --planner_node $PLANNER_NODE --actor_node $ACTOR_NODE
```
`$PLANNER_NODE` and `$ACTOR_NODE` should be manually set by the user (e.g., pool0-2838).

### 3-B. Run Data Collection Script for Parallel Processing
```bash
python parallel_collect_trajectories.py --planner_node $PLANNER_NODE --actor_node $ACTOR_NODE
```

## Running with SBATCH
TBD (we just need SBATCH script to run `parallel_collect_trajectories.py`)

