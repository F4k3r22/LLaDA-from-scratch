"""
This file is used to download a quantity x of files from the dataset and thus be able
to do training tests without having to download the entire 166GB dataset
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

def dw_dataset(repo_id: str = "Fredtt3/LLaDA-Sample-10BT", files_count: int = 10, path_to_dataset: str = "data_train"):
    repo_type = "dataset"
    out_dir   = Path(path_to_dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download processed_chunk_000000.pt through processed_chunk_000009.pt
    for idx in range(files_count):
        filename = f"processed_chunk_{idx:06d}.pt"
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            cache_dir=out_dir,
        )
        print("Downloaded to", path)
    
    return path.partition("/processed")[0]

if __name__ == "__main__":
    en = dw_dataset(files_count=2, path_to_dataset="data_train_en")
    print(en)
    #dw_dataset(repo_id="Fredtt3/LLaDA-Sample-ES", files_count=5, path_to_dataset="data_train_es")
    print("Done")