import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.preprocessing import Normalizer
from src.dataset import ThermalDataset
from src.model import ThermalAutoencoder

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocessing
    normalizer = Normalizer(os.path.join(cfg['paths']['metadata'], "stats.json"))
    print("Computing training statistics...")
    normalizer.fit(cfg['paths']['train_normal'])
    
    # Datasets
    aug_noise = cfg['training'].get('aug_noise_std', 0.0)
    train_dataset = ThermalDataset(cfg['paths']['train_normal'], normalizer, is_train=True, aug_noise_std=aug_noise)
    val_dataset = ThermalDataset(cfg['paths']['val_normal'], normalizer, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # Model
    model = ThermalAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # Training Loop
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    os.makedirs(cfg['paths']['models'], exist_ok=True)
    model_path = os.path.join(cfg['paths']['models'], "best_autoencoder.pth")
    
    print("Starting training...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Save history
    with open(os.path.join(cfg['paths']['metrics'], "loss_history.json"), 'w') as f:
        json.dump(history, f)
        
    print(f"Training completed. Best Val Loss: {best_val_loss:.6f}")
    
    # Reload best model for evaluation
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Print latent size
    latent_dummy = model.encoder(torch.zeros(1, 1, cfg['data']['grid_shape'][0], cfg['data']['grid_shape'][1]).to(device))
    latent_shape = latent_dummy.shape[1:]
    print(f"Latent size (channels x H x W): {latent_shape[0]} x {latent_shape[1]} x {latent_shape[2]}")
    
    # Compute error metrics on val set in denormalized units
    mae_list = []
    p99_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            for i in range(batch.size(0)):
                orig = normalizer.denormalize(batch[i, 0].cpu().numpy())
                rec = normalizer.denormalize(outputs[i, 0].cpu().numpy())
                err = np.abs(orig - rec)
                mae_list.append(np.mean(err))
                p99_list.append(np.percentile(err, 99))
                
    print(f"Validation MAE on NORMAL images: {np.mean(mae_list):.4f}")
    print(f"Validation 99th Percentile Error: {np.mean(p99_list):.4f}")

if __name__ == "__main__":
    main()
