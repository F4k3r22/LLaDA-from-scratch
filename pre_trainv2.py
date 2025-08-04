from model import LLaDAModel, LLaDAModelLM
from configs_llada import ModelConfig, LayerNormType, BlockType, InitFnType, ActivationType, ActivationCheckpointingStrategy, LLaDAConfig
import torch
from dataset import LLaDADatasetV2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from pathlib import Path
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

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

hf_configs = LLaDAConfig(d_model=768, n_heads=12, n_layers=14, 
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

def collate_fn_stack(batch):
    out = {}

    # 1) Apilar t (escalar)        
    t_vals = [sample["t"] for sample in batch]
    if not torch.is_tensor(t_vals[0]):
        t_vals = [torch.tensor(v, dtype=torch.float32) for v in t_vals]
    out["t"] = torch.stack(t_vals, dim=0)  # [B]

    # 2) Campos de secuencia
    seq_keys = ["input_ids", "noisy_input_ids", "mask"]
    # Descubre el pad_id (puedes ajustar según tu tokenizer/modelo)
    pad_id = tokenizer.pad_token_id if "tokenizer" in globals() else 0

    for key in seq_keys:
        seqs = [sample[key] for sample in batch]
        # Asegura que son tensores
        if not torch.is_tensor(seqs[0]):
            dtype = torch.long if key != "mask" else torch.bool
            seqs = [torch.tensor(s, dtype=dtype) for s in seqs]
        pad_val = False if key == "mask" else pad_id
        padded = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        out[key] = padded  # [B, L_max]

    return out

print("Load model test")
model = LLaDAModel(model_100M, init_params=True)
model.set_activation_checkpointing(ActivationCheckpointingStrategy.one_in_two)
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
hf_model = LLaDAModelLM(config=hf_configs, model=model)
print("Model test success")

dataset = LLaDADatasetV2(["/teamspace/studios/this_studio/data_train_en/datasets--Fredtt3--LLaDA-Sample-10BT/snapshots/ee6dbc7d4bf1e4b2d0974e48f5fcb8b62b1f27f4",
"/teamspace/studios/this_studio/data_train_es/datasets--Fredtt3--LLaDA-Sample-ES/snapshots/1f3128a94b4ff7e8d96892052704529e720f6b58"])
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_fn_stack
)


optimizer = AdamW(hf_model.parameters(), lr=4e-4, weight_decay=0.1)

total_steps = 50_000  
warmup_steps = 2_000
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

log_every  = 100
save_every = 500
output_dir = Path("checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)

for step, batch in enumerate(dataloader, start=1):
    hf_model.train()
    optimizer.zero_grad()

    # Batch now on device
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    inp   = batch["input_ids"]#.to(device)        # [B, L]
    noisy = batch["noisy_input_ids"]#.to(device)  # [B, L]
    mask  = batch["mask"]#.to(device)             # [B, L]
    t_vals= batch["t"]#.to(device)                # [B]

    # Sanity check prints
    if step % log_every == 0 or step == 1:
        print(f"\nStep {step}")
        print("→ t samples:", t_vals[:5].tolist())
        print("→ Masked ratios:", mask.float().mean(dim=1)[:5].tolist())


    # Forward
    logits = hf_model(noisy).logits                 # [B, L, V]

    # Loss diffusion: CE only on masked tokens, weighted 1/t
    B = inp.size(0)
    total_loss = 0.0
    for i in range(B):
        ti = t_vals[i]
        mi = mask[i]
        logits_i = logits[i, mi]              # [Ni, V]
        target_i = inp[i, mi]                 # [Ni]
        ce = F.cross_entropy(logits_i, target_i, reduction="sum")
        total_loss += ce / ti

    loss = total_loss / B

    # Backward + gradient clipping 
    loss.backward()
    # Grad norm check
    grad_norm = torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
    if step % log_every == 0 or step == 1:
        print(f"→ Grad norm: {grad_norm:.4f}")

    # Optimizer & scheduler step
    optimizer.step()
    scheduler.step()

    # Logging y checkpoints
    if step % log_every == 0:
        ppl = torch.exp(loss).item()
        print(f"[Step {step:6d}/{total_steps}] loss={loss.item():.4f} ppl={ppl:.2f}")

    if step % save_every == 0:
        checkpoint_dir = output_dir / f"llada_ckpt_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save the model in transformers format
        hf_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
