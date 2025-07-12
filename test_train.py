from model import LLaDAModel
from configs_llada import ModelConfig, LayerNormType, BlockType

# We will train and iterate from the base with a 100M parameter model to test and then scale to the 1B model.
# Ok for training we should use ModelConfig not LLaDAConfig
model_100M = ModelConfig(d_model=768, n_heads=12, n_layers=14, 
            n_kv_heads=12, mlp_ratio=4, mlp_hidden_size=3072,
            max_sequence_length=4096, vocab_size=126464,
            mask_token_id=126336, eos_token_id=126081,
            pad_token_id=126081, layer_norm_type=LayerNormType.rms,
            rms_norm_eps=1e-5, attention_dropout=0.0, residual_dropout=0.0,
            embedding_dropout=0.0, embedding_size=126464, block_type=BlockType.llama,
            block_group_size=1, attention_layer_norm=False, attention_layer_norm_with_affine=True,
            rope=True, rope_full_precision=True, rope_theta=500000.0, precision="bf16", weight_tying=False)

print("Load model test")
model = LLaDAModel(model_100M, init_params=True)
print("Model test success")


"""
According to the LLaDA paper, the training loop should:

- Sample t ∼ Uniform(0,1) for each chunk.
- Mask with mask = Bernoulli(t).
- Calculate the cross-entropy loss only on masked tokens and divide by t.
- Use AdamW (wd=0.1) and the Warmup–Stable–Decay scheduler as described (2000 steps warmup at 4e‑4; steady; decay to 1e‑4 after 1.2T tokens; final decay to 1e‑5 in the last 0.3T).
- Keep max_seq_length = 4096 and the 1% chunking variable in [1,4096].

https://arxiv.org/pdf/2502.09992 "Large Language Diffusion Models"
"""