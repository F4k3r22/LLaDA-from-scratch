from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import random
from tqdm import tqdm
import json
import os

## Just in case, I'll make a class to prepare data. I'll make it modular 
## and modifiable so that more people can understand it and adapt it to whatever is necessary.
## We will use datasets that are available in HF as a base

class PrepareData:
    def __init__(self, tokenizer: str = "GSAI-ML/LLaDA-8B-Instruct", 
        dataset_hf: str = "HuggingFaceFW/fineweb", sample: str = "sample-10BT", 
        max_seq_length: int = 4096, 
        id_mask_token: int = 126336, # According to the LLaDA paper and the official repo, the mask token is the one that has this ID :https://github.com/ML-GSAI/LLaDA/blob/main/app.py#L19
        num_proc: int = 8,
        output_dir: str = "data",
        chunks_per_file: int = 10000):

        self.tokenizer_name = tokenizer
        self.dataset_hf = dataset_hf
        self.sample = sample
        self.max_seq_length = max_seq_length
        self.id_mask_token = id_mask_token
        self.num_proc = num_proc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunks_per_file = chunks_per_file
        self.init_tokenizer()

    def init_tokenizer(self):
        """
        Initialize the tokenizer
        """
        print(f"Loading tokenizer from {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size

    def prepare_dataset(self, eps: float = 1e-3):
        print("==== Preparing Dataset ====")
        dataset = load_dataset(self.dataset_hf, name=self.sample, num_proc=self.num_proc)
        dataset = dataset['train']

        print("==== Tokenizing the entire dataset ====")
        device = torch.device("cpu")
        dataset = dataset.map(
            lambda x: {"input_ids": self.tokenizer(x["text"]).to(device)["input_ids"]},
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names
        )

        # Streaming processing: We accumulate tokens and process chunks when they reach the desired size
        print("==== Processing tokens in streaming mode ====")
        processed = []
        current_chunk = []  # Acumula los tokens
        chunk_count = 0
        file_count = 0


        # Define function to process each chunk: create tensor, mask and pack
        def process_chunk(chunk_tokens):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.int32, device=device) # We will use dtype=torch.int32 instead of torch.long for optimization reasons and because even with int32 we can load the entire vocab_size
            t = random.random()
            p_mask = (1.0 - eps) * t + eps
            mask = torch.rand(chunk_tensor.size(0), device=device) < p_mask
            noisy = chunk_tensor.clone()
            noisy[mask] = self.id_mask_token
            return {
                "t": t,
                "input_ids": chunk_tensor,
                "noisy_input_ids": noisy,
                "mask": mask
            }

        # Iterate over each example in the already tokenized dataset
        for example in tqdm(dataset, desc="Processing dataset"):
            tokens = example["input_ids"]
            current_chunk.extend(tokens)

            while len(current_chunk) >= self.max_seq_length:
                L = random.randint(1, self.max_seq_length) if random.random() < 0.01 else self.max_seq_length
                chunk_tokens = current_chunk[:L]
                processed.append(process_chunk(chunk_tokens))
                current_chunk = current_chunk[L:]
                chunk_count += 1

                # Save to file every N chunks
                if chunk_count % self.chunks_per_file == 0:
                    file_path = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
                    torch.save(processed, file_path)
                    print(f"✅ Saved {len(processed)} chunks to {file_path}")
                    processed = []
                    file_count += 1

        # Save what's left
        if processed:
            file_path = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
            torch.save(processed, file_path)
            print(f"✅ Saved final {len(processed)} chunks to {file_path}")

        print(f"==== Finished processing. Total chunks: {chunk_count} ====")

# Thanks to https://github.com/SpotDylan/llada-8b-fine-tune/blob/main/preprocess_alignment_data.py many of 
# the ideas and references to prepare data for SFT were taken from here, thank you very much in advance

class PrepareDataSFT:
    def __init__(
        self,
        data,
        tokenizer_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        max_seq_length: int = 4096,
        num_proc: int = 8,
        output_dir: str = "data"
    ):
        self.data = data
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def init_tokenizer(self):
        print(f"Loading tokenizer from {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True
        )
        self.vocab_size = self.tokenizer.vocab_size

    def prepare_and_save(self, output_file: str):
        """
        Process data into input_ids, attention_mask, and logits per token,
        then save as a single PT file.
        """
        assert hasattr(self, 'tokenizer'), "Call init_tokenizer() first"

        records = []
        for ex in tqdm(self.data, desc="Preparing data for SFT"):
            prompt = ex.get("prompt", "")
            response = ex.get("response", "")
            logits_data = ex.get("logits", [])

            # Tokenize separately to compute lengths
            enc_prompt = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            )
            enc_resp = self.tokenizer(
                response,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            )
            prompt_ids = enc_prompt.input_ids.squeeze(0)
            response_ids = enc_resp.input_ids.squeeze(0)

            combined_ids = torch.cat([prompt_ids, response_ids], dim=0)
            if combined_ids.size(0) > self.max_seq_length:
                continue  # skip too long

            # Prepare placeholder for logits
            llama_logits = torch.full(
                (combined_ids.size(0), self.vocab_size),
                fill_value=-100.0,
                dtype=torch.float32
            )

            # Align and fill logits
            L = prompt_ids.size(0)
            R = min(response_ids.size(0), len(logits_data))
            for i in range(R):
                entry = logits_data[i]
                pos = L + i
                chosen = entry.get("chosen_token_id")
                if chosen != response_ids[i].item():
                    print(f"Token mismatch at {pos}")
                if "full_logits" in entry:
                    fl = torch.tensor(entry["full_logits"], dtype=torch.float32)
                    if fl.numel() == self.vocab_size:
                        llama_logits[pos] = fl
                else:
                    top = entry.get("top_5", [])
                    for t in top:
                        tid = t.get("token_id")
                        if tid < self.vocab_size:
                            llama_logits[pos, tid] = t.get("logit")

            records.append({
                "input_ids": combined_ids,
                "prompt_length": L,
                "llama_logits": llama_logits
            })

        # Save all at once
        save_path = os.path.join(self.output_dir, output_file)
        torch.save(records, save_path)
        print(f"Saved {len(records)} examples to {save_path}")
                   

# This is for datasets that do NOT have logits, such as Hermes or others, 
# and we use this function to generate the logits for the SFT.
def compute_and_save_logits(dataset, tokenizer, model,
                            output_path, top_k=5):
    all_records = []

    for ex in tqdm(dataset, desc="Computing logits"):
        prompt, response = ex["prompt"], ex["response"]

        
        enc = tokenizer(prompt, response, return_tensors="pt")
        input_ids = enc.input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        
        with torch.no_grad():
            outputs = model(input_ids)
            
            logits = outputs.logits.squeeze(0)


        prompt_len   = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
        R = response_ids.shape[0]

       
        slice_start = prompt_len  
        token_logits = []
        for i in range(R):
            logit_vec = logits[slice_start + i]  

            if top_k is None:
                entry = {
                    "chosen_token_id": int(response_ids[i]),
                    "full_logits": logit_vec.cpu().tolist()
                }
            else:
                top = torch.topk(logit_vec, k=top_k)
                entry = {
                    "chosen_token_id": int(response_ids[i]),
                    "top_5": [
                        {"token_id": int(t), "logit": float(v)}
                        for t, v in zip(top.indices, top.values)
                    ]
                }
            token_logits.append(entry)

        record = {
            "prompt": prompt,
            "response": response,
            "logits": token_logits
        }
        all_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_records)} examples (with logits) to {output_path}")


# Silly example of how to clean and extract relevant information from the Hermes-3 Dataset
def hermes_dataset_prod(example):
    system_msgs = []
    human_msgs  = []
    gpt_msg     = ""

    
    for turn in example["conversations"]:
        role  = turn.get("from")
        text  = turn.get("value", "").strip()

        if role == "system":
            system_msgs.append(text)
        elif role == "human":
            human_msgs.append(text)
        elif role == "gpt":
            gpt_msg = text

    # Construye prompt y response
    prompt   = "\n".join(system_msgs + human_msgs)
    response = gpt_msg

    return {
        "prompt": prompt,
        "response": response
    }


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset

    model_name = "Fredtt3/LLaDA-100M-Test"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()


    ds = load_dataset("NousResearch/Hermes-3-Dataset")
    ds_train = ds["train"].map(
        hermes_dataset_prod,
        remove_columns=["conversations"],  
        batched=False                       
    )


    compute_and_save_logits(ds_train, tokenizer, model, output_path="data/data.json")

    with open(os.path.join("data", "data_with_logits.json"), "r", encoding="utf-8") as f:
        records = json.load(f)

    preparer = PrepareDataSFT(
        data=records,
        tokenizer_name=model_name,
        max_seq_length=4096,
        output_dir="data"
    )
    preparer.init_tokenizer()
    preparer.prepare_and_save(output_file="processed_data.pt")