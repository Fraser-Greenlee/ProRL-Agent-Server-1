# shellcheck shell=bash
# Qwen3.6-27B Megatron model args, shared by convert_weights.sh and run.sh.
#
# Qwen3.6-27B is a DENSE hybrid model: 3 GatedDeltaNet (linear) layers then 1
# full-attention layer, repeating (full_attention_interval=4) across 64 layers.
# Same architecture family as Qwen3.5-4B (the swegym example), so the same
# --spec applies; only the dimensions and a few flags differ.
#
# Differences from the 4B config worth flagging:
#   - tie_word_embeddings = FALSE (4B was true) → MUST add
#     --untie-embeddings-and-output-weights.
#   - 64 layers, hidden 5120, ffn 17408, 24 heads / 4 KV groups, head_dim 256.
#   - rope_theta 1e7, partial_rotary_factor 0.25 (rotary-percent 0.25).
#
# From text_config of Fraser/Qwen3.6-27B-ARC-Hy (== base Qwen/Qwen3.6-27B):
#   num_hidden_layers=64 hidden_size=5120 intermediate_size=17408
#   num_attention_heads=24 num_key_value_heads=4 head_dim=256
#   vocab_size=248320 rope_theta=10000000 rms_norm_eps=1e-6
#   partial_rotary_factor=0.25 attn_output_gate=true
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 24
    --num-query-groups 4
    --kv-channels 256
    --num-layers 64
    --hidden-size 5120
    --ffn-hidden-size 17408
    --use-gated-attention
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)
