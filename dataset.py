import os
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence, List, Tuple, Optional
from tqdm import tqdm

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
    def __init__(
        self,
        folder_paths: Union[str, Sequence[str]],
        max_files: Optional[int] = None
    ):
        """
        Dataset con índice y visualización de carga de archivos .pt.

        Parámetros:
        - folder_paths: ruta o lista de rutas a directorios con archivos .pt.
        - max_files: máximo de archivos .pt a indexar (None = todos).
        """
        if isinstance(folder_paths, str):
            paths = [folder_paths]
        else:
            paths = list(folder_paths)

        # Recolectar todos los nombres de archivos .pt
        all_files: List[str] = []
        for folder in paths:
            if not os.path.isdir(folder):
                raise ValueError(f"{folder!r} no es un directorio válido")
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.pt'):
                    all_files.append(os.path.join(folder, fname))

        # Aplicar límite si se especificó max_files
        if max_files is not None:
            files_to_index = all_files[:max_files]
        else:
            files_to_index = all_files

        self.index: List[Tuple[str, int]] = []
        total_files = len(files_to_index)

        # Barra de progreso para indexar archivos
        with tqdm(total=total_files, desc="Indexando archivos", unit="archivo") as pbar:
            for full_path in files_to_index:
                meta = torch.load(full_path, map_location="cpu")
                n = len(meta)
                # Añade cada chunk al índice
                self.index.extend([(full_path, i) for i in range(n)])
                pbar.update(1)

        print(f"Indexed {len(self.index)} samples across {total_files} files.")

        # Variables para cachear última carga
        self._cache_path: str = ""
        self._cache_data: List = []

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, local_idx = self.index[idx]
        # Mostrar al cargar un nuevo archivo durante iteración
        if path != self._cache_path:
            print(f"Loading chunk file: {os.path.basename(path)}")
            self._cache_data = torch.load(path, map_location="cpu")
            self._cache_path = path
        return self._cache_data[local_idx]
