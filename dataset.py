import os
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence


class LLaDADataset(Dataset):
    def __init__(self, folder_paths: Union[str, Sequence[str]], device: str):
        """
        folder_path: path to the folder containing one or more .pt files
        device: optional, device to move the tensioners to (cuda/cpu)
        """
        if isinstance(folder_paths, str):
            paths = [folder_paths]
        else:
            paths = list(folder_paths)

        self.samples = []
        self.device = device

        # Scan all .pt files in the folder
        for folder in paths:
            if not os.path.isdir(folder):
                raise ValueError(f"{folder!r} no es un directorio válido")
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".pt"):
                    full_path = os.path.join(folder, fname)
                    
                    data = torch.load(full_path, map_location=self.device)
                    self.samples.extend(data)

        print(f"Loaded {len(self.samples)} samples from {paths!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Move tensioners to the device if indicated
        if self.device is not None:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(self.device)
        return sample
