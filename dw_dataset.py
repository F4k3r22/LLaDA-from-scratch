"""
This file is used to download a quantity x of files from the dataset and thus be able 
to do training tests without having to download the entire 166GB dataset
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

repo_id   = "Fredtt3/LLaDA-Sample-10BT"
repo_type = "dataset"
out_dir   = Path("data_train")
out_dir.mkdir(parents=True, exist_ok=True)

# Download processed_chunk_000000.pt through processed_chunk_000009.pt
for idx in range(10):
    filename = f"processed_chunk_{idx:06d}.pt"
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        cache_dir=out_dir,
    )
    print("Downloaded to", path)