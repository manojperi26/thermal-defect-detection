import numpy as np
import json
import os
import torch

class Normalizer:
    def __init__(self, metadata_path="data/metadata/stats.json"):
        self.metadata_path = metadata_path
        self.mean = None
        self.std = None
        
    def fit(self, normal_train_dir):
        files = [f for f in os.listdir(normal_train_dir) if f.endswith('.npy')]
        all_data = []
        for f in files:
            arr = np.load(os.path.join(normal_train_dir, f))
            all_data.append(arr)
        
        all_data = np.stack(all_data)
        self.mean = float(np.mean(all_data))
        self.std = float(np.std(all_data))
        
        # Save stats
        with open(self.metadata_path, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)
            
    def load(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Stats file not found at {self.metadata_path}. Call fit() first.")
        with open(self.metadata_path, 'r') as f:
            stats = json.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            
    def normalize(self, img):
        return (img - self.mean) / self.std
        
    def denormalize(self, img):
        return img * self.std + self.mean

def add_gaussian_noise(tensor, std_dev=0.05):
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise
