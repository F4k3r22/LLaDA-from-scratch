from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import random
from tqdm import tqdm

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = dataset.map(
            lambda x: {"input_ids": self.tokenizer(x["text"]).to(device)["input_ids"]},
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names
        )

        # Procesamiento en streaming: acumulamos tokens y procesamos chunks cuando alcanzan el tamaño deseado
        print("==== Processing tokens in streaming mode ====")
        processed = []
        current_chunk = []  # Acumula los tokens
        chunk_count = 0
        file_count = 0


        # Define función para procesar cada chunk: crea tensor, enmascara y empaqueta
        def process_chunk(chunk_tokens):
            chunk_tensor = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
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

        # Itera sobre cada ejemplo del dataset ya tokenizado
        for example in tqdm(dataset, desc="Processing dataset"):
            tokens = example["input_ids"]
            current_chunk.extend(tokens)

            while len(current_chunk) >= self.max_seq_length:
                L = random.randint(1, self.max_seq_length) if random.random() < 0.01 else self.max_seq_length
                chunk_tokens = current_chunk[:L]
                processed.append(process_chunk(chunk_tokens))
                current_chunk = current_chunk[L:]
                chunk_count += 1

                # Guardar en archivo cada N chunks
                if chunk_count % self.chunks_per_file == 0:
                    file_path = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
                    torch.save(processed, file_path)
                    print(f"✅ Saved {len(processed)} chunks to {file_path}")
                    processed = []
                    file_count += 1

        # Guardar lo que queda
        if processed:
            file_path = self.output_dir / f"processed_chunk_{file_count:06d}.pt"
            torch.save(processed, file_path)
            print(f"✅ Saved final {len(processed)} chunks to {file_path}")

        print(f"==== Finished processing. Total chunks: {chunk_count} ====")



data = PrepareData(num_proc=32)

data.prepare_dataset()