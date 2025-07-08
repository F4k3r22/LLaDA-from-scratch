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

print("\n======== mask tokens =======\n")
# According to the LLaDA paper and the official repo, the mask token is the one that has this ID :https://github.com/ML-GSAI/LLaDA/blob/main/app.py#L19
token_id = 126336 # ID Mask
token_mask = tokenizer.convert_ids_to_tokens(token_id)
print("Mask token:", token_mask)        
print("Mask token ID:", token_id)   

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
