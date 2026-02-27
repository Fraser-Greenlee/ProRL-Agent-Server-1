 export HF_HUB_OFFLINE=1 && vllm serve MiniMaxAI/MiniMax-M2.1 --port 8000 --trust_remote_code --tensor-parallel-size 8 --enable_expert_parallel \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think

