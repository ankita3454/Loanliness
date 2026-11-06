# Experiment 1 - paper replication

Cleaning: 307,511 rows x 551 features -> 142,682 rows x 481 features (172 columns dropped for >30% missing).

## Balanced test set (comparable to paper TABLE III)
| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 70.34% | 0.71/0.70 | 0.70/0.71 | 0.70/0.71 | 0.770 | 0.763 |
| Random Forest | 69.38% | 0.70/0.69 | 0.68/0.71 | 0.69/0.70 | 0.761 | 0.749 |
| Naive Bayes | 62.55% | 0.62/0.63 | 0.65/0.60 | 0.63/0.62 | 0.655 | 0.625 |
| Multi-layer Perceptron | 68.04% | 0.72/0.65 | 0.59/0.77 | 0.65/0.71 | 0.753 | 0.740 |
| LightGBM | 70.14% | 0.71/0.70 | 0.69/0.71 | 0.70/0.70 | 0.768 | 0.764 |

## Natural (imbalanced) test set
| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 69.62% | 0.97/0.16 | 0.70/0.71 | 0.81/0.26 | 0.767 | 0.227 |
| Random Forest | 68.65% | 0.97/0.16 | 0.68/0.71 | 0.80/0.26 | 0.757 | 0.222 |
| Naive Bayes | 62.87% | 0.95/0.12 | 0.63/0.60 | 0.76/0.20 | 0.646 | 0.120 |
| Multi-layer Perceptron | 60.14% | 0.97/0.13 | 0.59/0.77 | 0.73/0.23 | 0.749 | 0.196 |
| LightGBM | 70.00% | 0.97/0.16 | 0.70/0.71 | 0.81/0.27 | 0.769 | 0.241 |

## Balancing strategies (natural test set)
| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | No balancing | 92.32% | 0.92/0.51 | 1.00/0.03 | 0.96/0.05 | 0.770 | 0.247 |
| Logistic Regression | Down-sampling | 69.62% | 0.97/0.16 | 0.70/0.71 | 0.81/0.26 | 0.767 | 0.227 |
| Logistic Regression | SMOTE | 71.92% | 0.96/0.17 | 0.72/0.67 | 0.83/0.27 | 0.754 | 0.234 |
| Logistic Regression | Class weights | 70.63% | 0.97/0.17 | 0.71/0.70 | 0.82/0.27 | 0.771 | 0.243 |
| LightGBM | No balancing | 92.33% | 0.93/0.52 | 1.00/0.03 | 0.96/0.06 | 0.770 | 0.247 |
| LightGBM | Down-sampling | 70.08% | 0.97/0.16 | 0.70/0.71 | 0.81/0.27 | 0.767 | 0.240 |
| LightGBM | SMOTE | 92.29% | 0.92/0.47 | 1.00/0.02 | 0.96/0.04 | 0.766 | 0.242 |
| LightGBM | Class weights | 74.24% | 0.96/0.18 | 0.75/0.66 | 0.84/0.28 | 0.772 | 0.256 |

## K-means (k=4) + per-cluster LightGBM (comparable to paper TABLE IV)
| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1 (0/1) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Cluster 1 | 71.38% | 0.74/0.68 | 0.74/0.69 | 0.74/0.68 | 0.762 | 0.701 |
| Cluster 2 | 68.57% | 0.72/0.64 | 0.72/0.64 | 0.72/0.64 | 0.739 | 0.705 |
| Cluster 3 | 69.39% | 0.61/0.74 | 0.53/0.79 | 0.57/0.76 | 0.736 | 0.801 |
| Cluster 4 | 67.81% | 0.70/0.66 | 0.68/0.68 | 0.69/0.67 | 0.729 | 0.696 |
| Overall | 69.18% | 0.70/0.69 | 0.68/0.71 | 0.69/0.70 | 0.751 | 0.739 |

Accuracy by k: k=2: 0.697, k=3: 0.692, k=4: 0.692, k=5: 0.686, k=6: 0.690
