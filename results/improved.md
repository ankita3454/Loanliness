# Experiment 2 - improved pipeline

307,511 applicants x 551 features, default rate 8.07%. 5-fold CV OOF ROC-AUC: **0.7899**. Threshold 0.16 picked on OOF predictions.

## Held-out test set (natural class balance)
| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LightGBM (all rows, all tables) | 86.30% | 0.95/0.28 | 0.90/0.46 | 0.92/0.35 | 0.794 | 0.298 |
| Logistic Regression (same data) | 71.26% | 0.96/0.18 | 0.71/0.70 | 0.82/0.28 | 0.775 | 0.261 |
| Demo model (20 typed-in fields) | 85.89% | 0.95/0.26 | 0.90/0.41 | 0.92/0.32 | 0.767 | 0.256 |

## Ablations (ROC-AUC)
| Setting | ROC-AUC |
|---|---|
| Full model | 0.7938 |
| No EXT_SOURCE scores | 0.7716 |
| Application table only | 0.7695 |
| Application only, no EXT_SOURCE | 0.7223 |

## Thin-file applicants (no Credit Bureau record)
```
{'n': 8776, 'default_rate': 0.10141294439380127, 'auc_full_model': 0.7779394289980538, 'auc_no_ext_source': 0.7652051566280167, 'auc_with_history': 0.7953548249770878}
```
