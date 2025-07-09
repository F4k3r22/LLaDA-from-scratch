from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import random

## Just in case, I'll make a class to prepare data. I'll make it modular 
## and modifiable so that more people can understand it and adapt it to whatever is necessary.
## We will use datasets that are available in HF as a base

class PrepareData:
    def __init__(self, tokenizer: str = "GSAI-ML/LLaDA-8B-Instruct", 
        dataset_hf: str = "HuggingFaceFW/fineweb", sample: str = "sample-10BT", 
        max_seq_length: int = 4096, 
        id_mask_token: int = 126336, # According to the LLaDA paper and the official repo, the mask token is the one that has this ID :https://github.com/ML-GSAI/LLaDA/blob/main/app.py#L19
        num_proc: int = 8,
        output_dir: str = "data",):

        self.tokenizer_name = tokenizer
        self.dataset_hf = dataset_hf
        self.sample = sample
        self.max_seq_length = max_seq_length
        self.id_mask_token = id_mask_token
        self.num_proc = num_proc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
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
        dataset = dataset.map(
            lambda x: {"input_ids": self.tokenizer(x["text"])["input_ids"]},
            num_proc=self.num_proc,
            remove_columns=dataset.column_names
        )

        print("==== Concatenating all tokens ====")
        all_ids = []
        for example in dataset:
            all_ids.extend(example["input_ids"])
        all_ids = torch.tensor(all_ids, dtype=torch.long)

        print("==== Dividing into chunks ====")
        chunks = []
        idx = 0
        N = len(all_ids)
        while idx < N:
            if random.random() < 0.01:
                L = random.randint(1, self.max_seq_length)
            else:
                L = self.max_seq_length
            chunk = all_ids[idx:idx+L]
            if chunk.size(0) < L:
                break
            chunks.append(chunk)
            idx += L

        print(f"==== Masking {len(chunks)} chunks ====")
        processed = []
        for chunk in chunks:
            # 1. Sample t ∈ [0,1]
            t = random.random()
            # 2. Compute p_mask with epsilon
            p_mask = (1.0 - eps) * t + eps
            # 3. Sample mask positions
            mask = torch.rand(chunk.size(0), device=chunk.device) < p_mask
            # 4. Create noisy input
            noisy = chunk.clone()
            noisy[mask] = self.id_mask_token
            # 5. Store also t for loss weighting
            processed.append({
                "t": t,
                "input_ids": chunk,
                "noisy_input_ids": noisy,
                "mask": mask
            })

        print("==== Saved ====")
        torch.save(processed, self.output_dir / "processed_dataset.pt")
        print(f"✅ Saved in {self.output_dir / 'processed_dataset.pt'}")



data = PrepareData()

data.prepare_dataset()