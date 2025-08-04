import os
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence, List, Tuple

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

class LLaDADatasetV2(Dataset):
    def __init__(self, folder_paths: Union[str, Sequence[str]]):
        if isinstance(folder_paths, str):
            paths = [folder_paths]
        else:
            paths = list(folder_paths)

        self.index: List[Tuple[str, int]] = []
        for folder in paths:
            if not os.path.isdir(folder):
                raise ValueError(f"{folder!r} no es un directorio válido")
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".pt"):
                    continue
                full_path = os.path.join(folder, fname)
                meta = torch.load(full_path, map_location="cpu")
                n = len(meta)
                self.index.extend([(full_path, i) for i in range(n)])

        print(f"Indexed {len(self.index)} samples across {len(paths)} folders.")

        # Para cachear
        self._cache_path: str  = ""
        self._cache_data: List = []

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, local_idx = self.index[idx]
        if path != self._cache_path:
            self._cache_data = torch.load(path, map_location="cpu")
            self._cache_path = path
        return self._cache_data[local_idx]