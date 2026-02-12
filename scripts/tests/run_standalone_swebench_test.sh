#!/bin/bash
#SBATCH --job-name=standalone-swebench-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000G
#SBATCH --partition=interactive
#SBATCH --time=4:00:00
#SBATCH --account=llmservice_fm_vision
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/haozh/outputs/verl_internal/results/slurm-%A_%a.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/haozh/outputs/verl_internal/results/slurm-%A_%a.err

set -x  # Enable debug output

# ==================== Configuration ====================
HOME_HAOZH='/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh'
OPENHANDS_WORKDIR=$HOME_HAOZH/projects/new_ProRL-Agent-Server/ProRL-Agent-Server
RESULTS_DIR="${OPENHANDS_WORKDIR}/results/standalone_test_$(date +%Y%m%d_%H%M%S)"
container_name="$HOME_HAOZH/singularity_images_v3/nvidian+nemo+verl_v2+vllm0.10dev.sqsh"
MOUNTS="--container-mounts=/lustre:/lustre"
# Model configuration
SFT_MODEL_PATH='/lustre/fsw/portfolios/llmservice/users/haozh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5'
TOKENIZER_PATH='/lustre/fsw/portfolios/llmservice/users/haozh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5'

# Data configuration
DATA_PATH="$HOME_HAOZH/data/swegym-new-split/test-transformed-with-prompt-first-64.parquet"
OUTPUT_DIR="${RESULTS_DIR}/standalone_swebench_test_${SLURM_JOB_ID}"

# Server configuration
GPUS_PER_NODE=8
TP_SIZE=4
GPU_MEM_UTIL=0.8
NUM_SERVERS_PER_NODE=$((GPUS_PER_NODE / TP_SIZE))
VLLM_BASE_PORT=8100
OPENHANDS_PORT=8006
OPENHANDS_NUM_WORKERS=64

# Evaluation configuration
NUM_TRAJECTORIES=1
TEMPERATURE=0.0
TOP_P=1.0
MAX_ITERATIONS=50
MAX_OUTPUT_TOKENS=1536
MAX_MODEL_LEN=32768
TIMEOUT=1500
HINT_MODE=none
TOKEN_LEVEL_GENERATION=true  # set to true for token-level generation

# ==================== Node Setup ====================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
NNODES=$SLURM_NNODES

mkdir -p "$RESULTS_DIR"

# ==================== Resolve Node IPs ====================
declare -a node_ips
for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    # Convert to ipv4 if needed
    if [[ "$node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            node_ip=${ADDR[1]}
        else
            node_ip=${ADDR[0]}
        fi
    fi
    node_ips[$i]=$node_ip
    echo "Node $i: ${nodes_array[$i]} -> IP: $node_ip"
done

# ==================== Start OpenHands on all nodes ====================
echo "Starting OpenHands servers on all nodes..."
openhands_urls=""

for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=${node_ips[$i]}

    echo "Starting OpenHands on node $node (IP: $node_ip)"

    srun --nodes=1 --ntasks=1 -w "$node" \
        -o "$RESULTS_DIR/output-%A_%a-openhands-node-$i.out" \
        -e "$RESULTS_DIR/output-%A_%a-openhands-node-$i.err" \
        --container-image="$container_name" $MOUNTS \
        bash -c "cd $OPENHANDS_WORKDIR \
        && export OH_RUNTIME_SINGULARITY_IMAGE_REPO=$HOME_HAOZH/singularity_images_v3 \
        && export OVERWRITE_OPENHANDS_DIR=$OPENHANDS_WORKDIR \
        && export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH \
        && export PYTHONPATH=$OPENHANDS_WORKDIR:\$PYTHONPATH \
        && export LOG_LEVEL=ERROR \
        && export DEBUG=False \
        && nohup /usr/bin/python scripts/start_server_thread.py --max-init-workers 70 --max-run-workers $OPENHANDS_NUM_WORKERS --timeout 9999999" &

    # Build the OpenHands URLs string
    if [ -z "$openhands_urls" ]; then
        openhands_urls="http://$node_ip:$OPENHANDS_PORT"
    else
        openhands_urls="$openhands_urls+http://$node_ip:$OPENHANDS_PORT"
    fi
done

echo "OpenHands URLs: $openhands_urls"

# ==================== Start vLLM servers on all nodes ====================
echo "Starting vLLM servers on all nodes..."
llm_server_urls=""

for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=${node_ips[$i]}

    echo "Starting $NUM_SERVERS_PER_NODE vLLM server(s) on node $node (IP: $node_ip)"

    for server_idx in $(seq 0 $((NUM_SERVERS_PER_NODE - 1))); do
        gpu_start=$((server_idx * TP_SIZE))
        gpu_end=$((gpu_start + TP_SIZE - 1))
        cuda_devices=$(seq -s, $gpu_start $gpu_end)
        port=$((VLLM_BASE_PORT + server_idx))

        if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
            # Token-level generation: use custom vllm_api_server.py
            vllm_cmd="CUDA_VISIBLE_DEVICES=$cuda_devices python $OPENHANDS_WORKDIR/scripts/tests/vllm_api_server.py \
                --model $SFT_MODEL_PATH \
                --tensor-parallel-size $TP_SIZE \
                --port $port \
                --host 0.0.0.0 \
                --gpu-memory-utilization $GPU_MEM_UTIL \
                --max-model-len $MAX_MODEL_LEN"
        else
            # Standard mode: use OpenAI-compatible vLLM server
            vllm_cmd="CUDA_VISIBLE_DEVICES=$cuda_devices python -m vllm.entrypoints.openai.api_server \
                --model $SFT_MODEL_PATH \
                --tensor-parallel-size $TP_SIZE \
                --port $port \
                --host 0.0.0.0 \
                --gpu-memory-utilization $GPU_MEM_UTIL \
                --max-model-len $MAX_MODEL_LEN"
        fi

        srun --nodes=1 --ntasks=1 -w "$node" \
            -o "$RESULTS_DIR/output-%A_%a-vllm-node-$i-server-$server_idx.out" \
            -e "$RESULTS_DIR/output-%A_%a-vllm-node-$i-server-$server_idx.err" \
            --container-image="$container_name" $MOUNTS \
            bash -c "$vllm_cmd" &

        # Build the LLM server URLs string
        if [ -z "$llm_server_urls" ]; then
            llm_server_urls="http://$node_ip:$port"
        else
            llm_server_urls="$llm_server_urls+http://$node_ip:$port"
        fi
    done
done

echo "LLM Server URLs: $llm_server_urls"

# ==================== Wait for servers to be ready ====================
echo "Waiting for servers to start up..."
sleep 120

# Health check for vLLM servers
echo "Checking vLLM server health..."
IFS='+' read -ra LLM_URLS <<< "$llm_server_urls"
for url in "${LLM_URLS[@]}"; do
    for attempt in $(seq 1 60); do
        if curl -s -o /dev/null -w "%{http_code}" "$url/health" | grep -q "200"; then
            echo "vLLM server $url is healthy"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo "WARNING: vLLM server $url did not become healthy after 5 minutes"
        fi
        sleep 5
    done
done

# ==================== Build evaluation command args ====================
TOKEN_LEVEL_FLAG=""
if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
    TOKEN_LEVEL_FLAG="--token_level_generation"
fi

# ==================== Run standalone evaluation ====================
echo "Starting standalone SWE-bench evaluation..."
echo "  OpenHands URLs: $openhands_urls"
echo "  LLM Server URLs: $llm_server_urls"

srun --overlap --nodes=1 --ntasks=1 -w "${nodes_array[0]}" \
    -o "$RESULTS_DIR/output-%A_%a-evaluation.out" \
    -e "$RESULTS_DIR/output-%A_%a-evaluation.err" \
    --container-image="$container_name" $MOUNTS \
    bash -c "cd $OPENHANDS_WORKDIR \
    && export PYTHONPATH=$OPENHANDS_WORKDIR:\$PYTHONPATH \
    && python scripts/tests/standalone_swebench_test.py \
        --data_path $DATA_PATH \
        --openhands_urls '$openhands_urls' \
        --llm_server_urls '$llm_server_urls' \
        --model_name $SFT_MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --num_trajectories $NUM_TRAJECTORIES \
        --num_workers_per_server $OPENHANDS_NUM_WORKERS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_iterations $MAX_ITERATIONS \
        --max_output_tokens $MAX_OUTPUT_TOKENS \
        --max_model_len $MAX_MODEL_LEN \
        --timeout $TIMEOUT \
        --hint_mode $HINT_MODE \
        --custom_tokenizer $TOKENIZER_PATH \
        $TOKEN_LEVEL_FLAG"

echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
