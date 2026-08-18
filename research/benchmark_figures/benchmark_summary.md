# TXM Crack Classifier -- Benchmark Report

Protocol: leave-one-image-out cross-validation across 4 images (LARGE_343_75, 333_75_um_zoom, 336_25, 338_13), 30000 pixels/class/image bootstrap sample.

## Model comparison (mean over 4 folds)

| Model | IoU | Dice | Precision | Recall | Fit time (s) | Predict time (s) | ROC AUC |
|---|---|---|---|---|---|---|---|
| RandomForest | 0.6841 | 0.8101 | 0.7443 | 0.8932 | 19.6 | 6.2 | 0.9550 |
| ExtraTrees | 0.6793 | 0.8068 | 0.7396 | 0.8930 | 3.4 | 11.3 | 0.9548 |
| HistGradientBoosting | 0.6923 | 0.8156 | 0.7495 | 0.9008 | 8.9 | 5.6 | 0.9542 |

## Figures

- `fig_a_model_comparison.png` -- accuracy bar chart
- `fig_b_area_fraction_parity.png` -- predicted vs actual crack coverage
- `fig_c_roc_curves.png` -- ROC curves
- `fig_d_confusion_matrix.png` -- confusion matrices
- `fig_e_feature_importance.png` -- Random Forest feature importance
- `fig_f_learning_curve.png` -- IoU vs training sample size
- `fig_g_decision_boundary.png` -- decision boundary, top-2 features
