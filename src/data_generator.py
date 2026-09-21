import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ThermalSimulation:
    """
    Conservative 3D heat diffusion model: dT/dt = 1/(rho*c) * div(k * grad(T))
    The object is a 3D grid (Z, Y, X) where Z is depth.
    Heat is applied via a Gaussian beam to the surface (Z=0).
    Boundaries: Heat flux at top surface, insulated everywhere else.
    """
    def __init__(self, cfg_physics, grid_shape, depth_layers):
        self.H, self.W = grid_shape
        self.D = depth_layers
        self.cfg_p = cfg_physics
        
        self.k_bg = cfg_physics['k_bg']
        self.rhoc_bg = cfg_physics['rhoc_bg']
        self.k_defect = cfg_physics['k_defect']
        self.rhoc_defect = cfg_physics['rhoc_defect']
        self.dt = cfg_physics['dt']
        self.timesteps_heat = cfg_physics['timesteps_heat']
        self.timesteps_observe = cfg_physics['timesteps_observe']
        
        # Enforce and assert FTCS stability condition
        k_eff = 2.0 * self.k_bg * self.k_defect / (self.k_bg + self.k_defect + 1e-8)
        alpha_eff = k_eff / min(self.rhoc_bg, self.rhoc_defect)
        alpha_max = max(self.k_bg / self.rhoc_bg, self.k_defect / self.rhoc_defect, alpha_eff)
        stability_val = alpha_max * self.dt
        assert stability_val <= 1.0 / 6.0, f"FTCS stability condition violated: {stability_val:.4f} > 1/6"
        
    def _create_material_grids(self, defect=None):
        k = np.full((self.D, self.H, self.W), self.k_bg, dtype=np.float32)
        rhoc = np.full((self.D, self.H, self.W), self.rhoc_bg, dtype=np.float32)
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        
        if defect is not None:
            d_type = defect['type']
            z = defect['depth']
            
            if d_type == 'circle':
                cy, cx = defect['center']
                r = defect['radius']
                Y, X = np.ogrid[:self.H, :self.W]
                dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
                defect_mask = dist_from_center <= r
                k[z:z+2, defect_mask] = self.k_defect
                rhoc[z:z+2, defect_mask] = self.rhoc_defect
                mask[defect_mask] = 1.0
                
            elif d_type == 'rectangle':
                y, x = defect['top_left']
                h, w = defect['h'], defect['w']
                k[z:z+2, y:y+h, x:x+w] = self.k_defect
                rhoc[z:z+2, y:y+h, x:x+w] = self.rhoc_defect
                mask[y:y+h, x:x+w] = 1.0
                
        return k, rhoc, mask

    def simulate(self, defect=None):
        k, rhoc, mask = self._create_material_grids(defect)
        
        # Randomize parameters for natural variation
        base_temp = np.random.normal(self.cfg_p['base_temp_mean'], self.cfg_p['base_temp_std'])
        intensity = max(1.0, np.random.normal(self.cfg_p['heat_intensity_mean'], self.cfg_p['heat_intensity_std']))
        beam_sigma = max(5.0, np.random.normal(self.cfg_p['beam_sigma_mean'], self.cfg_p['beam_sigma_std']))
        noise_std = max(0.001, np.random.normal(self.cfg_p['noise_std_mean'], self.cfg_p['noise_std_std']))
        
        # Initialize temperature
        T = np.full((self.D, self.H, self.W), base_temp, dtype=np.float32)
        
        # Gaussian heat beam
        Y, X = np.ogrid[:self.H, :self.W]
        cy, cx = self.H / 2.0, self.W / 2.0
        q_external = intensity * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * beam_sigma**2))
        
        for step in range(self.timesteps_observe):
            # Calculate fluxes using harmonic mean for thermal conductivity at interfaces
            # k_eff = 2 * k1 * k2 / (k1 + k2)
            # Add small epsilon to prevent division by zero just in case
            eps = 1e-8
            k_eff_x = 2 * k[:, :, :-1] * k[:, :, 1:] / (k[:, :, :-1] + k[:, :, 1:] + eps)
            q_x = np.zeros((self.D, self.H, self.W + 1), dtype=np.float32)
            q_x[:, :, 1:-1] = k_eff_x * (T[:, :, :-1] - T[:, :, 1:])
            net_in_x = q_x[:, :, :-1] - q_x[:, :, 1:]
            
            k_eff_y = 2 * k[:, :-1, :] * k[:, 1:, :] / (k[:, :-1, :] + k[:, 1:, :] + eps)
            q_y = np.zeros((self.D, self.H + 1, self.W), dtype=np.float32)
            q_y[:, 1:-1, :] = k_eff_y * (T[:, :-1, :] - T[:, 1:, :])
            net_in_y = q_y[:, :-1, :] - q_y[:, 1:, :]
            
            k_eff_z = 2 * k[:-1, :, :] * k[1:, :, :] / (k[:-1, :, :] + k[1:, :, :] + eps)
            q_z = np.zeros((self.D + 1, self.H, self.W), dtype=np.float32)
            q_z[1:-1, :, :] = k_eff_z * (T[:-1, :, :] - T[1:, :, :])
            
            # Apply heat source at surface (Z=0)
            if step < self.timesteps_heat:
                q_z[0, :, :] = q_external
            else:
                q_z[0, :, :] = 0.0 # Insulated during cooling
                
            net_in_z = q_z[:-1, :, :] - q_z[1:, :, :]
            
            # Temperature update (Conservative Heat Equation)
            T += (self.dt / rhoc) * (net_in_x + net_in_y + net_in_z)

        # Extract surface temperature
        surface_T = T[0, :, :]
        
        # Add sensor noise
        noise = np.random.normal(0, noise_std, surface_T.shape)
        surface_T += noise
        
        metadata = {
            'base_temp': float(base_temp),
            'heat_intensity': float(intensity),
            'beam_sigma': float(beam_sigma),
            'noise_std': float(noise_std)
        }
        
        return surface_T, mask, metadata

def generate_random_defect(H, W, D):
    defect_type = np.random.choice(['circle', 'rectangle'])
    # Depth from 1 to 4 (shallow=1-2, medium=3, deep=4)
    depth = np.random.randint(1, 5)
    
    if defect_type == 'circle':
        r = np.random.randint(3, 12)
        cy = np.random.randint(r, H - r)
        cx = np.random.randint(r, W - r)
        return {'type': 'circle', 'depth': depth, 'center': (int(cy), int(cx)), 'radius': int(r)}
    else:
        h = np.random.randint(6, 20)
        w = np.random.randint(6, 20)
        y = np.random.randint(0, H - h)
        x = np.random.randint(0, W - w)
        return {'type': 'rectangle', 'depth': depth, 'top_left': (int(y), int(x)), 'h': int(h), 'w': int(w)}

def save_sample(idx, img, mask, metadata, out_dir_img, out_dir_mask=None, prefix=""):
    np.save(os.path.join(out_dir_img, f"{prefix}_{idx:04d}.npy"), img)
    if out_dir_mask is not None:
        np.save(os.path.join(out_dir_mask, f"{prefix}_{idx:04d}.npy"), mask)
    
    return {
        'id': f"{prefix}_{idx:04d}",
        **metadata
    }

def main():
    cfg = load_config()
    np.random.seed(cfg['data']['seed'])
    
    # Ensure dirs exist
    for p in cfg['paths'].values():
        os.makedirs(p, exist_ok=True)
        
    sim = ThermalSimulation(cfg['data']['physics'], cfg['data']['grid_shape'], cfg['data']['depth_layers'])
    
    metadata_list = []
    
    # 1. Generate Train Normal
    print("Generating train normal...")
    for i in tqdm(range(cfg['data']['n_train_normal'])):
        img, mask, sample_meta = sim.simulate(defect=None)
        meta = save_sample(i, img, mask, {'split': 'train', 'label': 'normal', 'defect': None, **sample_meta}, cfg['paths']['train_normal'], prefix="normal")
        metadata_list.append(meta)
        
    # 2. Generate Val Normal
    print("Generating val normal...")
    for i in tqdm(range(cfg['data']['n_val_normal'])):
        img, mask, sample_meta = sim.simulate(defect=None)
        meta = save_sample(i, img, mask, {'split': 'val', 'label': 'normal', 'defect': None, **sample_meta}, cfg['paths']['val_normal'], prefix="normal")
        metadata_list.append(meta)
        
    # 3. Generate Test Normal
    print("Generating test normal...")
    for i in tqdm(range(cfg['data']['n_test_normal'])):
        img, mask, sample_meta = sim.simulate(defect=None)
        meta = save_sample(i, img, mask, {'split': 'test', 'label': 'normal', 'defect': None, **sample_meta}, cfg['paths']['test_normal'], prefix="normal")
        metadata_list.append(meta)
        
    # 4. Generate Test Defective
    print("Generating test defective...")
    for i in tqdm(range(cfg['data']['n_test_defective'])):
        defect = generate_random_defect(*cfg['data']['grid_shape'], cfg['data']['depth_layers'])
        img, mask, sample_meta = sim.simulate(defect=defect)
        meta = save_sample(i, img, mask, {'split': 'test', 'label': 'defective', 'defect': defect, **sample_meta}, cfg['paths']['test_defective'], cfg['paths']['masks'], prefix="defective")
        metadata_list.append(meta)
        
    # Save metadata
    with open(os.path.join(cfg['paths']['metadata'], 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata_list, f, indent=4)
        
    print("Generating smoke test visualization...")
    # Find a normal, a shallow defective, and a deep defective
    normal_img = np.load(os.path.join(cfg['paths']['train_normal'], "normal_0000.npy"))
    
    defects = [m for m in metadata_list if m['label'] == 'defective']
    defects.sort(key=lambda x: x['defect']['depth'])
    
    shallow = defects[0]
    deep = defects[-1]
    
    shallow_img = np.load(os.path.join(cfg['paths']['test_defective'], f"{shallow['id']}.npy"))
    shallow_mask = np.load(os.path.join(cfg['paths']['masks'], f"{shallow['id']}.npy"))
    
    deep_img = np.load(os.path.join(cfg['paths']['test_defective'], f"{deep['id']}.npy"))
    deep_mask = np.load(os.path.join(cfg['paths']['masks'], f"{deep['id']}.npy"))
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Normal
    im = axes[0,0].imshow(normal_img, cmap='inferno')
    axes[0,0].set_title("Normal (Raw Temp, Gaussian Heated)")
    fig.colorbar(im, ax=axes[0,0])
    axes[0,1].axis('off')
    axes[0,2].axis('off')
    
    # Shallow Defect
    im = axes[1,0].imshow(shallow_img, cmap='inferno')
    axes[1,0].set_title(f"Shallow Defect (depth={shallow['defect']['depth']})")
    fig.colorbar(im, ax=axes[1,0])
    
    im = axes[1,1].imshow(shallow_mask, cmap='gray')
    axes[1,1].set_title("GT Mask")
    
    # Deep Defect
    im = axes[2,0].imshow(deep_img, cmap='inferno')
    axes[2,0].set_title(f"Deep Defect (depth={deep['defect']['depth']})")
    fig.colorbar(im, ax=axes[2,0])
    
    im = axes[2,1].imshow(deep_mask, cmap='gray')
    axes[2,1].set_title("GT Mask")
    
    axes[1,2].axis('off')
    axes[2,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['paths']['figures'], 'smoke_test_generation.png'))
    plt.close()
    
    print("Phase 1 Data Generation Complete!")

if __name__ == "__main__":
    main()
