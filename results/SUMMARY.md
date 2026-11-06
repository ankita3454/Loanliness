# Results summary

## Paper vs our replication (accuracy on a balanced test set)

| Model | Paper | Ours | Our ROC-AUC (natural test) |
|---|---|---|---|
| Logistic Regression | 69.34% | 70.34% | 0.767 |
| Random Forest | 63.51% | 69.38% | 0.757 |
| Naive Bayes | 52.11% | 62.55% | 0.646 |
| Multi-layer Perceptron | 69.15% | 68.04% | 0.749 |
| LightGBM | 57.47% | 70.14% | 0.769 |
| K-means (k=4) + per-cluster LightGBM | 71.57% | 69.18% | - |

## Improved pipeline (natural test set, all applicants)

* Rows used: paper-style cleaning keeps 142,682 of 307,511; improved pipeline keeps all 307,511.
* Best paper-style model ROC-AUC: 0.769
* Improved LightGBM ROC-AUC: **0.794** (CV OOF 0.790), PR-AUC 0.298 vs base rate 0.081
* At threshold 0.16: catches 46% of defaults with 28% precision; balanced accuracy 67.7%

## Ablations (ROC-AUC)

* Full model: 0.794
* No EXT_SOURCE scores: 0.772
* Application table only: 0.769
* Application only, no EXT_SOURCE: 0.722

## Thin-file applicants (no Credit Bureau record)

* 8,776 test applicants, default rate 10.1%
* ROC-AUC 0.778 (vs 0.795 with history); without external scores 0.765
