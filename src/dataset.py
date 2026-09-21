import os
import torch
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing import Normalizer, add_gaussian_noise

class ThermalDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer, is_train=False, aug_noise_std=0.0):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer
        self.is_train = is_train
        self.aug_noise_std = aug_noise_std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        img = np.load(file_path).astype(np.float32)
        
        # Normalize
        img = self.normalizer.normalize(img)
        
        # Convert to tensor (C, H, W) -> (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)
        
        if self.is_train and self.aug_noise_std > 0:
            tensor = add_gaussian_noise(tensor, self.aug_noise_std)
            
        return tensor
