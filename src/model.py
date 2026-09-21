import torch
import torch.nn as nn

class ThermalAutoencoder(nn.Module):
    def __init__(self):
        super(ThermalAutoencoder, self).__init__()
        
        # Encoder
        # Input: 1 x 64 x 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 64 x 8 x 8 (Bottleneck)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 32 x 16 x 16
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # 16 x 32 x 32
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)   # 1 x 64 x 64
            # No activation at the end to allow arbitrary values
        )
        
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
