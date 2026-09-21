import os
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from skimage import morphology
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import ThermalAutoencoder
from baseline import BaselineModel
from preprocessing import Normalizer

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_gt_mask(info, H=64, W=64):
    mask = np.zeros((H, W), dtype=bool)
    if not info.get('defect', False) or info['defect'] is None: return mask
    
    dt = info['defect']['type']
    if dt == 'rectangle':
        y1, x1 = info['defect']['top_left']
        h, w = info['defect']['h'], info['defect']['w']
        y2 = min(H, y1 + h)
        x2 = min(W, x1 + w)
        mask[y1:y2, x1:x2] = True
    else: # circle
        cy, cx = info['defect']['center']
        r = info['defect']['radius']
        Y, X = np.ogrid[:H, :W]
        dist = (Y - cy)**2 + (X - cx)**2
        mask[dist <= r**2] = True
    return mask

def compute_localization(err_map, threshold):
    binary = err_map > threshold
    binary = morphology.opening(binary, morphology.disk(1))
    binary = morphology.closing(binary, morphology.disk(2))
    return binary

def calc_iou_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0
    
    tp = intersection
    fp = pred_mask.sum() - tp
    fn = gt_mask.sum() - tp
    px_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    px_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return iou, dice, px_prec, px_rec

def process_dataset(data_dir, meta_file, model, baseline, normalizer, device, split_name, label_name):
    with open(meta_file, 'r') as f:
        full_meta = json.load(f)
        
    meta = [m for m in full_meta if m['split'] == split_name and m['label'] == label_name]
    results = []
    
    for i, info in enumerate(meta):
        filename = f"{info['id']}.npy"
            
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, f"{info['id']}.npy")
            if not os.path.exists(file_path):
                file_path = os.path.join(data_dir, f"defective_{info['id']}.npy")

        img_orig = np.load(file_path).astype(np.float32)
        img = normalizer.normalize(img_orig)
        
        # AE inference
        tensor_img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon_ae = model(tensor_img).squeeze().cpu().numpy()
        
        mse_ae = np.mean((img - recon_ae)**2)
        err_ae = np.abs(img - recon_ae)
        
        # Denormalized error
        recon_ae_orig = normalizer.denormalize(recon_ae)
        err_ae_orig = np.abs(img_orig - recon_ae_orig)
        
        # Baseline inference
        recon_base = baseline.predict(img)
        mse_base = np.mean((img - recon_base)**2)
        err_base = np.abs(img - recon_base)
        
        recon_base_orig = normalizer.denormalize(recon_base)
        err_base_orig = np.abs(img_orig - recon_base_orig)
        
        results.append({
            'info': info,
            'mse_ae': mse_ae,
            'err_ae': err_ae,
            'err_ae_orig': err_ae_orig,
            'mse_base': mse_base,
            'err_base': err_base,
            'err_base_orig': err_base_orig,
            'gt_mask': get_gt_mask(info, *img.shape)
        })
        
    return results

def evaluate_metrics(results, img_threshold, px_threshold, prefix):
    y_true = np.array([1 if r['info'].get('defect', False) else 0 for r in results])
    y_scores = np.array([r[f'mse_{prefix}'] for r in results])
    y_pred = np.array([1 if s > img_threshold else 0 for s in y_scores])
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0
    
    # Bootstrap CI for AUC
    auc_ci_lower = 0.0
    auc_ci_upper = 0.0
    if len(set(y_true)) > 1:
        n_boot = 1000
        boot_aucs = []
        n = len(y_true)
        for _ in range(n_boot):
            indices = np.random.choice(n, n, replace=True)
            if len(set(y_true[indices])) > 1:
                boot_aucs.append(roc_auc_score(y_true[indices], y_scores[indices]))
        if boot_aucs:
            auc_ci_lower = np.percentile(boot_aucs, 2.5)
            auc_ci_upper = np.percentile(boot_aucs, 97.5)
    
    ious, dices, px_precs, px_recs = [], [], [], []
    for r, is_defective, pred in zip(results, y_true, y_pred):
        if is_defective and pred == 1:
            pred_mask = compute_localization(r[f'err_{prefix}'], px_threshold)
            iou, dice, p_prec, p_rec = calc_iou_dice(pred_mask, r['gt_mask'])
            ious.append(iou)
            dices.append(dice)
            px_precs.append(p_prec)
            px_recs.append(p_rec)
            
    return {
        'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'fpr': float(fpr),
        'auc': float(auc), 'auc_ci_lower': float(auc_ci_lower), 'auc_ci_upper': float(auc_ci_upper),
        'iou': float(np.mean(ious)) if ious else 0.0,
        'dice': float(np.mean(dices)) if dices else 0.0,
        'px_precision': float(np.mean(px_precs)) if px_precs else 0.0,
        'px_recall': float(np.mean(px_recs)) if px_recs else 0.0,
        'y_scores': [float(s) for s in y_scores],
        'y_true': [int(y) for y in y_true]
    }

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer(os.path.join(cfg['paths']['metadata'], "stats.json"))
    normalizer.load()
    model = ThermalAutoencoder().to(device)
    model.load_state_dict(torch.load(os.path.join(cfg['paths']['models'], "best_autoencoder.pth"), map_location=device, weights_only=True))
    model.eval()
    baseline = BaselineModel()
    
    print("Processing Val Normal...")
    meta_path = os.path.join(cfg['paths']['metadata'], "dataset_metadata.json")
    val_res = process_dataset(cfg['paths']['val_normal'], meta_path, model, baseline, normalizer, device, 'val', 'normal')
    
    # AE Thresholds
    val_mse_ae = [r['mse_ae'] for r in val_res]
    val_px_ae = np.concatenate([r['err_ae'].flatten() for r in val_res])
    ae_px_99 = np.percentile(val_px_ae, 99)
    ae_img_mean3std = np.mean(val_mse_ae) + 3*np.std(val_mse_ae)
    ae_img_95 = np.percentile(val_mse_ae, 95)
    ae_img_99 = np.percentile(val_mse_ae, 99)
    
    # Base Thresholds
    val_mse_base = [r['mse_base'] for r in val_res]
    val_px_base = np.concatenate([r['err_base'].flatten() for r in val_res])
    base_px_99 = np.percentile(val_px_base, 99)
    base_img_mean3std = np.mean(val_mse_base) + 3*np.std(val_mse_base)
    base_img_95 = np.percentile(val_mse_base, 95)
    base_img_99 = np.percentile(val_mse_base, 99)
    
    # Denormalized Validation Error
    err_ae_orig = np.concatenate([r['err_ae_orig'].flatten() for r in val_res])
    mae_ae_orig = np.mean(err_ae_orig)
    p99_ae_orig = np.percentile(err_ae_orig, 99)
    
    print(f"AE Normal Validation Error (Denormalized): MAE={mae_ae_orig:.4f} °C, 99th={p99_ae_orig:.4f} °C")
    
    print("Processing Test Normal & Defective...")
    test_norm_res = process_dataset(cfg['paths']['test_normal'], meta_path, model, baseline, normalizer, device, 'test', 'normal')
    test_def_res = process_dataset(cfg['paths']['test_defective'], meta_path, model, baseline, normalizer, device, 'test', 'defective')
    
    all_test = test_norm_res + test_def_res
    
    # Global metrics using 99th percentile for general storage
    met_ae = evaluate_metrics(all_test, ae_img_99, ae_px_99, 'ae')
    met_base = evaluate_metrics(all_test, base_img_99, base_px_99, 'base')
    
    # Multi-threshold per-depth classification
    depths = sorted(list(set([r['info']['defect']['depth'] for r in test_def_res])))
    depth_metrics = []
    
    for d in depths:
        def_subset = [r for r in test_def_res if r['info']['defect']['depth'] == d]
        subset = test_norm_res + def_subset
        
        # Evaluate for all 3 thresholds
        res = {'depth': d, 'n_defective': len(def_subset), 'n_normal': len(test_norm_res)}
        
        # AE
        res['ae'] = {
            'mean3std': evaluate_metrics(subset, ae_img_mean3std, ae_px_99, 'ae'),
            'p95': evaluate_metrics(subset, ae_img_95, ae_px_99, 'ae'),
            'p99': evaluate_metrics(subset, ae_img_99, ae_px_99, 'ae')
        }
        
        # Base
        res['base'] = {
            'mean3std': evaluate_metrics(subset, base_img_mean3std, base_px_99, 'base'),
            'p95': evaluate_metrics(subset, base_img_95, base_px_99, 'base'),
            'p99': evaluate_metrics(subset, base_img_99, base_px_99, 'base')
        }
        
        # Remove y_scores/y_true to save space
        for t in ['mean3std', 'p95', 'p99']:
            for m in ['ae', 'base']:
                res[m][t].pop('y_scores', None)
                res[m][t].pop('y_true', None)
                
        depth_metrics.append(res)
            
    results_dict = {
        'global': {
            'ae': met_ae,
            'baseline': met_base
        },
        'per_depth': depth_metrics,
        'thresholds': {
            'ae_img_mean3std': float(ae_img_mean3std), 'ae_img_95': float(ae_img_95), 'ae_img_99': float(ae_img_99), 'ae_px_99': float(ae_px_99),
            'base_img_mean3std': float(base_img_mean3std), 'base_img_95': float(base_img_95), 'base_img_99': float(base_img_99), 'base_px_99': float(base_px_99)
        },
        'val_orig_error': {
            'mae': float(mae_ae_orig),
            'p99': float(p99_ae_orig)
        }
    }
    
    os.makedirs(cfg['paths']['metrics'], exist_ok=True)
    with open(os.path.join(cfg['paths']['metrics'], "evaluation_fixed.json"), 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print("Metrics saved to results/metrics/evaluation_fixed.json")

if __name__ == "__main__":
    main()
