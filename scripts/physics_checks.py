import os
import yaml
import numpy as np
import scipy.ndimage as ndimage
from src.data_generator import ThermalSimulation

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_checks():
    cfg = load_config()
    
    # Check 1: Depth table
    print("--- Check 1: Depth Table (Fixed seed, Noise OFF) ---")
    cfg_fixed = cfg['data']['physics'].copy()
    cfg_fixed['noise_std_mean'] = 0.0
    cfg_fixed['noise_std_std'] = 0.0
    cfg_fixed['base_temp_std'] = 0.0
    cfg_fixed['heat_intensity_std'] = 0.0
    cfg_fixed['beam_sigma_std'] = 0.0
    
    sim = ThermalSimulation(cfg_fixed, cfg['data']['grid_shape'], cfg['data']['depth_layers'])
    
    # First, get a purely normal image as baseline
    np.random.seed(42)
    normal_img, _, _ = sim.simulate(defect=None)
    
    depths = range(1, 9)
    contrasts = []
    
    for d in depths:
        defect = {'type': 'rectangle', 'depth': d, 'top_left': (20, 20), 'h': 10, 'w': 10}
        np.random.seed(42) # Keep random heating params exactly the same
        def_img, mask, _ = sim.simulate(defect=defect)
        
        # Anomaly is difference from normal
        anomaly = def_img - normal_img
        
        # Contrast: Mean in defect region vs surrounding
        surrounding_mask = ndimage.binary_dilation(mask, iterations=3) ^ mask.astype(bool)
        
        mean_defect = np.mean(anomaly[mask.astype(bool)])
        mean_surround = np.mean(anomaly[surrounding_mask])
        contrast = np.abs(mean_defect - mean_surround)
        contrasts.append(contrast)
        
        # Edge sharpness: Gradient magnitude at the edges
        grad_y, grad_x = np.gradient(anomaly)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        edge_sharpness = np.mean(grad_mag[surrounding_mask])
        
        print(f"Depth {d}: Contrast = {contrast:.4f}, Edge Sharpness = {edge_sharpness:.4f}")
        
    print("\n--- Check 2: Signal vs Normal Variation ---")
    sim_noisy = ThermalSimulation(cfg['data']['physics'], cfg['data']['grid_shape'], cfg['data']['depth_layers'])
    normal_images = []
    for i in range(100):
        np.random.seed(1000 + i)
        img, _, _ = sim_noisy.simulate(defect=None)
        normal_images.append(img)
    
    normal_images = np.array(normal_images)
    pixel_std = np.std(normal_images, axis=0)
    mean_normal_variation = np.mean(pixel_std)
    
    print(f"Typical Defect Contrast (Depth 1): {contrasts[0]:.4f}")
    print(f"Typical Defect Contrast (Depth 4): {contrasts[3]:.4f}")
    print(f"Mean Normal Image Variation (std per pixel across 100 samples): {mean_normal_variation:.4f}")
    print(f"Expected Noise Std: {cfg['data']['physics']['noise_std_mean']:.4f}")
    print(f"Signal-to-Variation Ratio (Depth 1): {contrasts[0]/mean_normal_variation:.2f}")
    print(f"Signal-to-Noise Ratio (Depth 1): {contrasts[0]/cfg['data']['physics']['noise_std_mean']:.2f}")
    if mean_normal_variation > contrasts[0]:
        print("WARNING: Normal variation is LARGER than the strongest defect signal!")
    
    print("\n--- Check 3: Energy Conservation ---")
    k, rhoc, _ = sim._create_material_grids(defect=None)
    T = np.full((sim.D, sim.H, sim.W), cfg_fixed['base_temp_mean'], dtype=np.float32)
    Y, X = np.ogrid[:sim.H, :sim.W]
    cy, cx = sim.H / 2.0, sim.W / 2.0
    q_external = cfg_fixed['heat_intensity_mean'] * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * cfg_fixed['beam_sigma_mean']**2))
    
    eps = 1e-8
    k_eff_x = 2 * k[:, :, :-1] * k[:, :, 1:] / (k[:, :, :-1] + k[:, :, 1:] + eps)
    k_eff_y = 2 * k[:, :-1, :] * k[:, 1:, :] / (k[:, :-1, :] + k[:, 1:, :] + eps)
    k_eff_z = 2 * k[:-1, :, :] * k[1:, :, :] / (k[:-1, :, :] + k[1:, :, :] + eps)

    heats = []
    for step in range(sim.timesteps_observe):
        q_x = np.zeros((sim.D, sim.H, sim.W + 1), dtype=np.float32)
        q_x[:, :, 1:-1] = k_eff_x * (T[:, :, :-1] - T[:, :, 1:])
        net_in_x = q_x[:, :, :-1] - q_x[:, :, 1:]
        
        q_y = np.zeros((sim.D, sim.H + 1, sim.W), dtype=np.float32)
        q_y[:, 1:-1, :] = k_eff_y * (T[:, :-1, :] - T[:, 1:, :])
        net_in_y = q_y[:, :-1, :] - q_y[:, 1:, :]
        
        q_z = np.zeros((sim.D + 1, sim.H, sim.W), dtype=np.float32)
        q_z[1:-1, :, :] = k_eff_z * (T[:-1, :, :] - T[1:, :, :])
        
        if step < sim.timesteps_heat:
            q_z[0, :, :] = q_external
        else:
            q_z[0, :, :] = 0.0 # Insulated during cooling
            
        net_in_z = q_z[:-1, :, :] - q_z[1:, :, :]
        
        T += (sim.dt / rhoc) * (net_in_x + net_in_y + net_in_z)
        total_heat = np.sum(rhoc * T)
        heats.append(total_heat)
    
    cooling_heats = heats[sim.timesteps_heat:]
    if len(cooling_heats) > 0:
        drift = cooling_heats[-1] - cooling_heats[0]
        max_diff = np.max(cooling_heats) - np.min(cooling_heats)
        print(f"Total heat at start of cooling: {cooling_heats[0]:.4f}")
        print(f"Total heat at end of cooling:   {cooling_heats[-1]:.4f}")
        print(f"Drift during cooling phase:     {drift:.8e} (Max Diff: {max_diff:.8e})")
    else:
        print("No cooling steps defined.")
        
    print("\n--- Check 4: Stability & Alignment ---")
    k_eff_val = 2.0 * cfg['data']['physics']['k_bg'] * cfg['data']['physics']['k_defect'] / (cfg['data']['physics']['k_bg'] + cfg['data']['physics']['k_defect'] + eps)
    alpha_eff = k_eff_val / min(cfg['data']['physics']['rhoc_bg'], cfg['data']['physics']['rhoc_defect'])
    alpha_max = max(cfg['data']['physics']['k_bg']/cfg['data']['physics']['rhoc_bg'], 
                    cfg['data']['physics']['k_defect']/cfg['data']['physics']['rhoc_defect'],
                    alpha_eff)
    dx = 1.0 # explicitly defined
    stability_val = alpha_max * cfg['data']['physics']['dt'] / (dx**2)
    print(f"Stability value (alpha * dt / dx^2): {stability_val:.4f} (Must be <= 0.1667)")
    
    defect = {'type': 'circle', 'depth': 1, 'center': (30, 45), 'radius': 5}
    np.random.seed(42)
    def_img, mask, _ = sim.simulate(defect=defect)
    np.random.seed(42)
    norm_img, _, _ = sim.simulate(defect=None)
    anomaly = def_img - norm_img
    
    cy_a, cx_a = ndimage.center_of_mass(np.abs(anomaly))
    cy_m, cx_m = ndimage.center_of_mass(mask)
    print(f"Ground Truth Mask Center: y={cy_m:.1f}, x={cx_m:.1f}")
    print(f"Surface Anomaly Center:   y={cy_a:.1f}, x={cx_a:.1f}")
    
if __name__ == '__main__':
    run_checks()
