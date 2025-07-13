from model import LLaDAModel
from configs_llada import ModelConfig, LayerNormType, BlockType, InitFnType, ActivationType
import torch
from dataset import LLaDADataset
from torch.utils.data import DataLoader

# We will train and iterate from the base with a 100M parameter model to test and then scale to the 1B model.
# Ok for training we should use ModelConfig not LLaDAConfig
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
model_100M = ModelConfig(d_model=768, n_heads=12, n_layers=14, 
            n_kv_heads=12, mlp_ratio=4, mlp_hidden_size=3072,
            max_sequence_length=4096, vocab_size=126464,
            mask_token_id=126336, eos_token_id=126081,
            pad_token_id=126081, layer_norm_type=LayerNormType.rms,
            rms_norm_eps=1e-5, attention_dropout=0.0, residual_dropout=0.0,
            embedding_dropout=0.0, embedding_size=126464, block_type=BlockType.llama,
            block_group_size=1, attention_layer_norm=False, attention_layer_norm_with_affine=True,
            rope=True, rope_full_precision=True, rope_theta=500000.0, precision="bf16", weight_tying=False,
            init_device=device, init_fn=InitFnType.mitchell, init_std=0.02, activation_type=ActivationType.swiglu,
            alibi=False, alibi_bias_max=8.0)

print("Load model test")
model = LLaDAModel(model_100M, init_params=True)
print("Model test success")

dataset = LLaDADataset("data_train/datasets--Fredtt3--LLaDA-Sample-10BT", device=device)
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

"""
According to the LLaDA paper, the training loop should:

- Sample t ∼ Uniform(0,1) for each chunk.
- Mask with mask = Bernoulli(t).
- Calculate the cross-entropy loss only on masked tokens and divide by t. (sum)
- Use AdamW (wd=0.1) and the Warmup–Stable–Decay scheduler as described (2000 steps warmup at 4e‑4; steady; decay to 1e‑4 after 1.2T tokens; final decay to 1e‑5 in the last 0.3T).
- Keep max_seq_length = 4096 and the 1% chunking variable in [1,4096].

https://arxiv.org/pdf/2502.09992 "Large Language Diffusion Models"
"""