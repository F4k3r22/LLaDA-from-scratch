from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path

## Just in case, I'll make a class to prepare data. I'll make it modular 
## and modifiable so that more people can understand it and adapt it to whatever is necessary.
## We will use datasets that are available in HF as a base

class PrepareData:
    def __init__(self, tokenizer: str = "GSAI-ML/LLaDA-8B-Instruct", 
        dataset_hf: str = "HuggingFaceFW/fineweb", sample: str = "sample-10BT", 
        max_seq_length: int = 4096, id_mask_token: int = 126336,
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
        print(f"Cargando tokenizer desde {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size

    def procces(self, example):
        pass