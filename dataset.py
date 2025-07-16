import os
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence

class LLaDADataset(Dataset):
    def __init__(self, folder_paths: Union[str, Sequence[str]]):
        """
        Carga todos los .pt en RAM (CPU) al crear el dataset.
        No toca la GPU en esta fase.
        """
        if isinstance(folder_paths, str):
            paths = [folder_paths]
        else:
            paths = list(folder_paths)

        self.samples = []

        for folder in paths:
            if not os.path.isdir(folder):
                raise ValueError(f"{folder!r} no es un directorio válido")
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".pt"):
                    full_path = os.path.join(folder, fname)
                    # Se carga siempre en CPU
                    data = torch.load(full_path, map_location="cpu")
                    self.samples.extend(data)

        print(f"Loaded {len(self.samples)} samples from {paths!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Devuelve todo en CPU
        return self.samples[idx]