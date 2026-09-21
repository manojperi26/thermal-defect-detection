import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import Normalizer
from src.data_generator import ThermalSimulation
from src.model import ThermalAutoencoder
from src.evaluate import compute_localization

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def plot_loss(cfg):
    history_path = os.path.join(cfg['paths']['metrics'], "loss_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (Log Scale)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cfg['paths']['figures'], "loss_curve.png"))
        plt.close()
        print("Loss curve saved.")

def plot_sample_simulation(cfg):
    np.random.seed(42)
    grid_shape = tuple(cfg['data']['grid_shape'])
    sim = ThermalSimulation(cfg['data']['physics'], grid_shape, cfg['data']['depth_layers'])
    
    # We will simulate normal, shallow, and deep defect with exactly the same seed states
    # To do this, we save the numpy RNG state and restore it for each call
    rng_state = np.random.get_state()
    
    # 1. Normal
    np.random.set_state(rng_state)
    img_normal, _, _ = sim.simulate(defect=None)
    
    # 2. Shallow (Depth 1)
    np.random.set_state(rng_state)
    defect_shallow = {'type': 'circle', 'depth': 1, 'center': (32, 32), 'radius': 5}
    img_shallow, _, _ = sim.simulate(defect=defect_shallow)
    _, _, mask_shallow = sim._create_material_grids(defect_shallow)
    
    # 3. Deep (Depth 4)
    np.random.set_state(rng_state)
    defect_deep = {'type': 'circle', 'depth': 4, 'center': (32, 32), 'radius': 5}
    img_deep, _, _ = sim.simulate(defect=defect_deep)
    _, _, mask_deep = sim._create_material_grids(defect_deep)
    
    vmin = min(img_normal.min(), img_shallow.min(), img_deep.min())
    vmax = max(img_normal.max(), img_shallow.max(), img_deep.max())
    
    diff_shallow = img_shallow - img_normal
    diff_deep = img_deep - img_normal
    v_diff = max(abs(diff_shallow).max(), abs(diff_deep).max(), 1e-5)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Row 0: Normal
    im_img = axes[0, 0].imshow(img_normal, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Normal Image')
    im_diff = axes[0, 1].imshow(np.zeros_like(img_normal), cmap='coolwarm', vmin=-v_diff, vmax=v_diff)
    axes[0, 1].set_title('Diff (Defect - Normal)')
    axes[0, 2].imshow(np.zeros_like(img_normal), cmap='gray')
    axes[0, 2].set_title('GT Mask')
    
    # Row 1: Shallow
    axes[1, 0].imshow(img_shallow, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Shallow Defect (Depth 1)')
    axes[1, 1].imshow(diff_shallow, cmap='coolwarm', vmin=-v_diff, vmax=v_diff)
    axes[1, 1].set_title('Diff (Shallow)')
    axes[1, 2].imshow(mask_shallow, cmap='gray')
    axes[1, 2].set_title('GT Mask')
    
    # Row 2: Deep
    axes[2, 0].imshow(img_deep, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[2, 0].set_title('Deep Defect (Depth 4)')
    axes[2, 1].imshow(diff_deep, cmap='coolwarm', vmin=-v_diff, vmax=v_diff)
    axes[2, 1].set_title('Diff (Deep)')
    axes[2, 2].imshow(mask_deep, cmap='gray')
    axes[2, 2].set_title('GT Mask')
    
    # Add a single colorbar for all difference panels
    cbar = fig.colorbar(im_diff, ax=axes[:, 1].ravel().tolist(), label='Difference (°C)', fraction=0.046, pad=0.04)
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('results/figures/sample_simulation.png')
    plt.close()
    print("Sample simulation saved.")

def plot_localization_overlays(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder().to(device)
    model.load_state_dict(torch.load(os.path.join(cfg['paths']['models'], "best_autoencoder.pth"), map_location=device, weights_only=True))
    model.eval()
    
    normalizer = Normalizer(os.path.join(cfg['paths']['metadata'], "stats.json"))
    normalizer.load()
    
    with open('results/metrics/evaluation_fixed.json', 'r') as f:
        metrics = json.load(f)
    px_threshold = metrics['thresholds']['ae_px_99']
    
    meta_path = os.path.join(cfg['paths']['metadata'], "dataset_metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    test_defective = [m for m in meta if m['split'] == 'test' and m['label'] == 'defective']
    
    # Pick 2 shallow (depth 1 or 2), 2 medium (depth 3), 2 deep (depth 4)
    samples = {'shallow': [], 'medium': [], 'deep': []}
    for m in test_defective:
        d = m['defect']['depth']
        if d in [1, 2] and len(samples['shallow']) < 2:
            samples['shallow'].append(m)
        elif d == 3 and len(samples['medium']) < 2:
            samples['medium'].append(m)
        elif d == 4 and len(samples['deep']) < 2:
            samples['deep'].append(m)
            
    selected_meta = samples['shallow'] + samples['medium'] + samples['deep']
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    for i, info in enumerate(selected_meta):
        row = i // 2
        col = i % 2
        
        filepath = os.path.join(cfg['paths']['test_defective'], f"defective_{info['id']}.npy")
        if not os.path.exists(filepath):
            filepath = os.path.join(cfg['paths']['test_defective'], f"{info['id']}.npy")
            
        img_orig = np.load(filepath).astype(np.float32)
        img = normalizer.normalize(img_orig)
        
        tensor_img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(tensor_img).squeeze().cpu().numpy()
            
        err = np.abs(img - recon)
        pred_mask = compute_localization(err, px_threshold)
        
        # GT mask
        from src.evaluate import get_gt_mask
        gt_mask = get_gt_mask(info)
        
        ax = axes[row, col]
        ax.imshow(img_orig, cmap='inferno')
        
        # Overlay pred_mask
        ax.contour(pred_mask, levels=[0.5], colors=['red'], linewidths=2, alpha=0.7)
        # Overlay GT mask
        ax.contour(gt_mask, levels=[0.5], colors=['lime'], linewidths=1.5, linestyles='dashed')
        
        d_type = 'Shallow' if row == 0 else ('Medium' if row == 1 else 'Deep')
        ax.set_title(f"{d_type} (Depth {info['defect']['depth']})")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('results/figures/localization_overlays.png')
    plt.close()
    print("Localization overlays saved.")

from sklearn.metrics import roc_curve, auc, roc_auc_score
from src.baseline import BaselineModel

def plot_roc_and_depth(cfg):
    with open('results/metrics/evaluation_fixed.json', 'r') as f:
        data = json.load(f)
        
    y_true_ae = data['global']['ae']['y_true']
    y_scores_ae = data['global']['ae']['y_scores']
    
    y_true_base = data['global']['baseline']['y_true']
    y_scores_base = data['global']['baseline']['y_scores']
    
    fpr_ae, tpr_ae, _ = roc_curve(y_true_ae, y_scores_ae)
    fpr_base, tpr_base, _ = roc_curve(y_true_base, y_scores_base)
    
    # Partial AUC up to FPR=0.1
    idx_ae = np.where(fpr_ae <= 0.1)[0]
    pauc_ae = auc(fpr_ae[idx_ae], tpr_ae[idx_ae])
    
    idx_base = np.where(fpr_base <= 0.1)[0]
    pauc_base = auc(fpr_base[idx_base], tpr_base[idx_base])
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_ae, tpr_ae, label=f"AE (AUC={data['global']['ae']['auc']:.3f}, pAUC@0.1={pauc_ae:.3f})")
    plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC={data['global']['baseline']['auc']:.3f}, pAUC@0.1={pauc_base:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/roc_curve.png')
    plt.close()
    
    # Performance vs depth plot
    depths = sorted([d['depth'] for d in data['per_depth']])
    
    ae_auc = [d['ae']['p99']['auc'] for d in data['per_depth']]
    ae_auc_lower = [d['ae']['p99']['auc'] - d['ae']['p99']['auc_ci_lower'] for d in data['per_depth']]
    ae_auc_upper = [d['ae']['p99']['auc_ci_upper'] - d['ae']['p99']['auc'] for d in data['per_depth']]
    
    base_auc = [d['base']['p99']['auc'] for d in data['per_depth']]
    base_auc_lower = [d['base']['p99']['auc'] - d['base']['p99']['auc_ci_lower'] for d in data['per_depth']]
    base_auc_upper = [d['base']['p99']['auc_ci_upper'] - d['base']['p99']['auc'] for d in data['per_depth']]
    
    ae_rec_95 = [d['ae']['p95']['recall'] for d in data['per_depth']]
    ae_f1_95 = [d['ae']['p95']['f1'] for d in data['per_depth']]
    base_rec_95 = [d['base']['p95']['recall'] for d in data['per_depth']]
    base_f1_95 = [d['base']['p95']['f1'] for d in data['per_depth']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): AUC with 95% CI error bars
    x = np.arange(len(depths))
    width = 0.3
    
    axes[0].bar(x - width/2, ae_auc, width, yerr=[ae_auc_lower, ae_auc_upper], capsize=5, label='Autoencoder')
    axes[0].bar(x + width/2, base_auc, width, yerr=[base_auc_lower, base_auc_upper], capsize=5, label='Baseline')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(depths)
    axes[0].set_xlabel('Defect Depth')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('(a) Classification AUC by Depth')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Panel (b): Recall and F1 at 95th Percentile
    axes[1].plot(depths, ae_rec_95, 'o-', label='AE Recall')
    axes[1].plot(depths, ae_f1_95, 's--', label='AE F1')
    axes[1].plot(depths, base_rec_95, 'o-', label='Baseline Recall')
    axes[1].plot(depths, base_f1_95, 's--', label='Baseline F1')
    
    axes[1].set_xlabel('Defect Depth')
    axes[1].set_ylabel('Score')
    axes[1].set_title('(b) Metrics at 95th Percentile Threshold')
    axes[1].set_xticks(depths)
    axes[1].legend()
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('results/figures/performance_vs_depth.png')
    plt.close()
    print("ROC and Depth Performance plots saved.")

def process_dataset_with_noise(data_dir, meta_file, model, baseline, normalizer, device, split_name, label_name, sigma):
    with open(meta_file, 'r') as f:
        full_meta = json.load(f)
        
    meta = [m for m in full_meta if m['split'] == split_name and m['label'] == label_name]
    
    y_true = []
    y_scores_ae = []
    y_scores_base = []
    
    for i, info in enumerate(meta):
        filename = f"{info['id']}.npy"
            
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, f"{info['id']}.npy")
            if not os.path.exists(file_path):
                file_path = os.path.join(data_dir, f"defective_{info['id']}.npy")

        img_orig = np.load(file_path).astype(np.float32)
        img = normalizer.normalize(img_orig)
        
        # ADD NOISE
        if sigma > 0:
            img = img + np.random.normal(0, sigma, img.shape).astype(np.float32)
        
        # AE inference
        tensor_img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon_ae = model(tensor_img).squeeze().cpu().numpy()
        
        mse_ae = np.mean((img - recon_ae)**2)
        
        # Baseline inference
        recon_base = baseline.predict(img)
        mse_base = np.mean((img - recon_base)**2)
        
        is_defect = 1 if info.get('defect', False) else 0
        
        y_true.append(is_defect)
        y_scores_ae.append(mse_ae)
        y_scores_base.append(mse_base)
        
    return y_true, y_scores_ae, y_scores_base

def plot_noise_robustness(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer(os.path.join(cfg['paths']['metadata'], "stats.json"))
    normalizer.load()
    model = ThermalAutoencoder().to(device)
    model.load_state_dict(torch.load(os.path.join(cfg['paths']['models'], "best_autoencoder.pth"), map_location=device, weights_only=True))
    model.eval()
    baseline = BaselineModel()
    
    meta_path = os.path.join(cfg['paths']['metadata'], "dataset_metadata.json")
    
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    ae_aucs = []
    base_aucs = []
    
    for sigma in sigmas:
        y_true_n, ae_s_n, base_s_n = process_dataset_with_noise(cfg['paths']['test_normal'], meta_path, model, baseline, normalizer, device, 'test', 'normal', sigma)
        y_true_d, ae_s_d, base_s_d = process_dataset_with_noise(cfg['paths']['test_defective'], meta_path, model, baseline, normalizer, device, 'test', 'defective', sigma)
        
        y_true = np.array(y_true_n + y_true_d)
        ae_s = np.array(ae_s_n + ae_s_d)
        base_s = np.array(base_s_n + base_s_d)
        
        auc_ae = roc_auc_score(y_true, ae_s)
        auc_base = roc_auc_score(y_true, base_s)
        
        ae_aucs.append(auc_ae)
        base_aucs.append(auc_base)

    plt.figure(figsize=(7, 5))
    sigmas_c = [s * normalizer.std for s in sigmas]
    plt.plot(sigmas_c, ae_aucs, 'o-', label='Autoencoder')
    plt.plot(sigmas_c, base_aucs, 's--', label='Baseline')
    
    # Mark native noise level
    native_noise = cfg['data']['physics']['noise_std_mean']
    plt.axvline(x=native_noise, color='r', linestyle=':', label='Native Noise Level')
    
    plt.xlabel('Added Noise (Std Dev, °C)')
    plt.ylabel('Global ROC-AUC')
    plt.title('Robustness to Additional Sensor Noise')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/figures/noise_robustness.png')
    plt.close()
    print("Noise robustness plot saved.")

def main():
    cfg = load_config()
    os.makedirs(cfg['paths']['figures'], exist_ok=True)
    
    plot_loss(cfg)
    plot_sample_simulation(cfg)
    plot_localization_overlays(cfg)
    plot_roc_and_depth(cfg)
    plot_noise_robustness(cfg)

if __name__ == "__main__":
    main()
