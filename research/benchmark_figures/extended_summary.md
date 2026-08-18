# Extended comparisons -- neural network + interpretability tiers

## Neural network, same LOIO protocol as fig_a

| Model | IoU | Dice | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| RandomForest | 0.6841 | 0.8101 | 0.7443 | 0.8932 | 0.9550 |
| ExtraTrees | 0.6793 | 0.8068 | 0.7396 | 0.8930 | 0.9548 |
| HistGradientBoosting | 0.6923 | 0.8156 | 0.7495 | 0.9008 | 0.9542 |
| MLP (neural network) | 0.7343 | 0.8465 | 0.8071 | 0.8903 | 0.9680 |

Neural network fit+predict wall time: 59s (4 folds, hidden_layer_sizes=[64, 32]).

## Interpretability tiers

| Tier | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| Otsu threshold (classical) | 0.4258 | 0.5941 | 0.5577 | 0.7124 |
| Logistic regression (identified equation) | 0.5478 | 0.7076 | 0.6568 | 0.7849 |
| Decision tree (depth 3) | 0.5813 | 0.7334 | 0.6429 | 0.8693 |
| HistGradientBoosting (full ML) | 0.6923 | 0.8156 | 0.7495 | 0.9008 |

**Identified equation**: `logit(P(crack)) = +3.351 +6.580×gradmag_s8 +43.670×texture_s8 -4.903×smooth_s2 -2.392×intensity`

