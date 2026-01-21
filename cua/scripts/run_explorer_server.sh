vllm serve /lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --limit-mm-per-prompt.video 0 \
  --limit-mm-per-prompt.image 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
