from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import random


def dataset_finetuning(example, input_ids: str = "problem", output_ids: str = "generated_solution", mask_token_id: int = 126336):
    tokenized_prompt = tokenizer(example[f"{input_ids}"], truncation=True, padding=False)
    tokenized_resp   = tokenizer(example[f"{output_ids}"], truncation=True, padding=False)

    input_ids  = tokenized_prompt["input_ids"] + tokenized_resp["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_resp["input_ids"]
    t = random.random()

    mask_id = mask_token_id
    for i in range(len(tokenized_prompt["input_ids"]), len(input_ids)):
        if random.random() < t:
            labels[i] = mask_id  # castigamos al modelo a predecir mask_token

    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.int32),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int32),
        "labels":         torch.tensor(labels, dtype=torch.int32),
    }

device = 'cuda'
model = AutoModel.from_pretrained('Fredtt3/LLaDA-100M-Test', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('Fredtt3/LLaDA-100M-Test', trust_remote_code=True)
ds = load_dataset("XenArcAI/MathX-5M")

train_ds = ds.map(dataset_finetuning, remove_columns=ds.column_names)
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

lora_conf = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
peft_model = get_peft_model(model, lora_conf)

args = TrainingArguments(
    output_dir="./lora_llada_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    fp16=True,
    max_steps=2000,
    logging_steps=50,
    save_steps=500,
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_ds,
    
)
trainer.train()

peft_model.save_pretrained("./lora_sft_adapters")
tokenizer.save_pretrained("./lora_sft_adapters")