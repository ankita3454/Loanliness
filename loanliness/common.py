"""Metrics, result saving and plotting helpers shared by all experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)

# Categorical palette (fixed order) + neutral inks
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REPAID, DEFAULT = COLORS[0], COLORS[1]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Metrics in the paper's format (per-class precision/recall/F1) plus AUCs."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_0": p[0], "precision_1": p[1],
        "recall_0": r[0], "recall_1": r[1],
        "f1_0": f[0], "f1_1": f[1],
        "threshold": threshold,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    return {k: float(v) for k, v in out.items()}


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 of the default class (chosen on validation data only)."""
    grid = np.linspace(0.02, 0.98, 97)
    scores = []
    for t in grid:
        _, _, f, _ = precision_recall_fscore_support(y_true, (proba >= t).astype(int),
                                                     labels=[1], zero_division=0)
        scores.append(f[0])
    return float(grid[int(np.argmax(scores))])


def paper_style_table(rows: dict[str, dict]) -> pd.DataFrame:
    """Format like the paper's TABLE III: 'class0/class1' strings + AUC columns."""
    recs = []
    for name, m in rows.items():
        recs.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.2%}",
            "Precision (0/1)": f"{m['precision_0']:.2f}/{m['precision_1']:.2f}",
            "Recall (0/1)": f"{m['recall_0']:.2f}/{m['recall_1']:.2f}",
            "F1 (0/1)": f"{m['f1_0']:.2f}/{m['f1_1']:.2f}",
            "ROC-AUC": f"{m.get('roc_auc', float('nan')):.3f}",
            "PR-AUC": f"{m.get('pr_auc', float('nan')):.3f}",
        })
    return pd.DataFrame(recs)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def savefig(fig, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
