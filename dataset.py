import os
import torch
from torch.utils.data import Dataset

class LLaDADataset(Dataset):
    def __init__(self, folder_path: str, device: str):
        """
        folder_path: path to the folder containing one or more .pt files
        device: optional, device to move the tensioners to (cuda/cpu)
        """
        self.samples = []
        self.device = device

        # Scan all .pt files in the folder
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".pt"):
                full_path = os.path.join(folder_path, fname)
                # Each .pt must contain a list of dicts: [{ "t":..., "input_ids":..., ... }, ...]
                data = torch.load(full_path, map_location=self.device)
                self.samples.extend(data)

        print(f"Loaded {len(self.samples)} samples from {folder_path}")

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
