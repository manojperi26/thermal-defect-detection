# ThermoRecon: Unsupervised Thermal Defect Detection Using Autoencoders

This project implements an unsupervised anomaly detection system for localizing subsurface defects in materials using active thermal imaging (thermography) and Convolutional Autoencoders.

## Project Overview

In active thermography, a material is heated, and its surface temperature is monitored over time. Subsurface defects alter the heat diffusion, causing surface temperature anomalies. This project uses a Physics Engine (3D finite-difference heat diffusion) to simulate these anomalies. We then train a Convolutional Autoencoder (AE) **only on defect-free samples** and detect anomalies based on the reconstruction error. We compare the AE against a simple polynomial fitting baseline.

## Running the Code

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate dataset & Run Training:**
   ```bash
   python src/data_generator.py
   python src/train.py
   ```
3. **Evaluate and Plot Results:**
   ```bash
   python src/evaluate.py
   python src/visualize.py
   ```
4. **Print Metrics Summary:**
   ```bash
   python scripts/print_summary.py
   ```

## Results Summary

The Autoencoder performs approximately the same as the simple polynomial fitting baseline overall. There is no claim that the AE is strictly better; rather, they have complementary strengths depending on the defect depth.

- **Global AE AUC:** 0.829 (95% CI: 0.787-0.869) | F1: 0.592
- **Global Baseline AUC:** 0.825 (95% CI: 0.779-0.864) | F1: 0.462

### Performance by Depth

*Evaluation metrics computed using the 99th Percentile Threshold. Normal validation error (Denormalized) was MAE 0.1262 °C with a 99th Percentile pixel error of 0.4668 °C.*

| Depth | n(Def) | Model | Prec  | Rec   | F1    | FPR   | AUC (95% CI)         | IoU(TP)|
|-------|--------|-------|-------|-------|-------|-------|----------------------|--------|
| 1     | 42     | AE    | 1.000 | 0.976 | 0.988 | 0.000 | 1.000 (0.998-1.000)  | 0.153  |
| 1     | 42     | BASE  | 0.921 | 0.833 | 0.875 | 0.015 | 0.996 (0.990-0.999)  | 0.181  |
| 2     | 57     | AE    | 1.000 | 0.754 | 0.860 | 0.000 | 0.993 (0.984-0.998)  | 0.263  |
| 2     | 57     | BASE  | 0.889 | 0.421 | 0.571 | 0.015 | 0.974 (0.955-0.989)  | 0.530  |
| 3     | 41     | AE    | 0.000 | 0.000 | 0.000 | 0.000 | 0.827 (0.753-0.888)  | 0.000  |
| 3     | 41     | BASE  | 0.000 | 0.000 | 0.000 | 0.015 | 0.805 (0.733-0.867)  | 0.000  |
| 4     | 60     | AE    | 0.000 | 0.000 | 0.000 | 0.000 | 0.557 (0.476-0.643)  | 0.000  |
| 4     | 60     | BASE  | 0.400 | 0.033 | 0.062 | 0.015 | 0.579 (0.494-0.660)  | 0.114  |

## Limitations & Future Work

- **Parity with Baseline**: The AE ≈ baseline overall. While the AE slightly wins on image-level classification for shallow depths (Depth 1 & 2 AUC/Recall), the Baseline wins significantly on pixel-level localization precision (IoU) for Depth 2, providing a much cleaner mask without the AE's blurring effect.
- **Deep Defects at Chance**: Depth 4 classification is at chance level (AUC ~0.55-0.58). Both models fail to threshold any defects at Depth 3 and 4 using the 99th percentile because the reconstruction error for normal sample variations (noise, beam intensity) exceeds the weak thermal contrast signal.
- **Noise Robustness**: The baseline is MORE robust than the AE to added noise (AE AUC drops from 0.83 to 0.61 at max noise, while the baseline drops from 0.83 to 0.77). A key limitation is that the AE was trained at a single noise level, which may explain its brittleness compared to the baseline fitting.
