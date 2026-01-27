vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --api-key gen \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.image 5 \
  --limit-mm-per-prompt.video 0 \
  --max-model-len 65536
