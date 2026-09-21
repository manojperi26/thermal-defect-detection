import json

with open('results/metrics/evaluation_fixed.json', 'r') as f:
    d = json.load(f)

print('--- GLOBAL VALIDATION RECONSTRUCTION ERROR (Denormalized) ---')
val = d['val_orig_error']
print(f"MAE: {val['mae']:.4f} °C, 99th Percentile Pixel Error: {val['p99']:.4f} °C\n")


print('--- GLOBAL THRESHOLDS (MSE) ---')
for k, v in d['thresholds'].items():
    print(f"{k}: {v:.6f}")
print()

def print_table(thresh_key, title):
    print(f'--- {title} ---')
    header = f"{'Depth':<6} | {'n(Def)':<6} | {'Model':<4} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FPR':<6} | {'AUC (95% CI)':<22} | {'IoU(TP)':<7}"
    print(header)
    print("-" * len(header))
    for r in d['per_depth']:
        n_def = r['n_defective']
        flag = "*" if n_def < 10 else " "
        depth_str = f"{r['depth']}{flag}"
        
        for model in ['ae', 'base']:
            m = r[model][thresh_key]
            auc_str = f"{m['auc']:.3f} ({m['auc_ci_lower']:.3f}-{m['auc_ci_upper']:.3f})"
            print(f"{depth_str:<6} | {n_def:<6} | {model.upper():<4} | {m['precision']:<6.3f} | {m['recall']:<6.3f} | {m['f1']:<6.3f} | {m['fpr']:<6.3f} | {auc_str:<22} | {m['iou']:<7.3f}")
    print()

print_table('mean3std', 'PER-DEPTH METRICS: Threshold = Mean + 3*Std')
print_table('p95', 'PER-DEPTH METRICS: Threshold = 95th Percentile')
print_table('p99', 'PER-DEPTH METRICS: Threshold = 99th Percentile')

print('--- GLOBAL TEST CLASSIFICATION ---')
ae = d['global']['ae']
base = d['global']['baseline']
print(f"AE AUC: {ae['auc']:.3f} ({ae['auc_ci_lower']:.3f}-{ae['auc_ci_upper']:.3f}), F1: {ae['f1']:.3f}")
print(f"Base AUC: {base['auc']:.3f} ({base['auc_ci_lower']:.3f}-{base['auc_ci_upper']:.3f}), F1: {base['f1']:.3f}")
