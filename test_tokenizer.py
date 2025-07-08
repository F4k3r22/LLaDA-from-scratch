from transformers import AutoTokenizer
import torch

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

prompt = "Test for tokenizer"

# 1. Mira todos los tokens especiales que conoce el tokenizer:
print("\n======== tokens maps =======\n")
print(tokenizer.special_tokens_map)           
print("\n======== tokens list =======\n")
print(tokenizer.all_special_tokens)         
print("\n======== tokens ids =======\n")
print(tokenizer.all_special_ids)              
print("\n")

# 2. Inspecciona la cadena final tras apply_chat_template:
wrapped = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False
)
print("Cadena con template:", wrapped)

# 3. Tokeniza en detalle (token y su ID):
for tok, idx in zip(tokenizer.tokenize(wrapped), tokenizer(wrapped)["input_ids"]):
    print(f"{tok:>15} → {idx}")
