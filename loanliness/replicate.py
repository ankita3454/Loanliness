"""Experiment 1: replicate the Loanliness paper's pipeline.

Mirrors the paper (Sections IV-V):
  * drop columns with >30% missing values, then drop rows that still have any NaN
  * label-encode binary categoricals, one-hot encode the rest
  * degree-2 polynomial features on a handful of numeric columns
  * down-sample the majority class, normalise, train LR / RF / NB / MLP / LightGBM
  * compare balancing strategies (none, down-sampling, SMOTE, class weights)
  * K-means clustering + one LightGBM per cluster (paper TABLE IV)

The paper reports metrics on a *balanced* test set. We report that (to compare
with the paper) AND the natural, imbalanced test set (what a lender would see).

Usage:  python -m loanliness.replicate
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config
from .common import (COLORS, evaluate, paper_style_table, plt, save_json, savefig, to_markdown)
from .features import categorical_columns, clean_names, feature_columns, load_features


# ----------------------------------------------------------- paper preprocessing
def paper_clean(df: pd.DataFrame, threshold: float = config.PAPER_MISSING_THRESHOLD):
    """Return (X, y, report) following the paper's cleaning recipe."""
    feats = feature_columns(df)
    before = {"rows": len(df), "features": len(feats)}

    missing = df[feats].isna().mean()
    keep = missing[missing <= threshold].index.tolist()
    data = df[keep + ["TARGET"]].dropna()

    cats = [c for c in categorical_columns(df) if c in keep]
    binary = [c for c in cats if data[c].nunique() <= 2]
    multi = [c for c in cats if c not in binary]
    for c in binary:
        data[c] = data[c].astype("category").cat.codes
    data = pd.get_dummies(data, columns=multi, dtype=np.float32)

    poly_cols = [c for c in config.PAPER_POLY_COLUMNS if c in data]
    if len(poly_cols) >= 2:
        pf = PolynomialFeatures(2, include_bias=False)
        poly = pf.fit_transform(data[poly_cols].to_numpy(dtype=np.float64))
        names = pf.get_feature_names_out(poly_cols)
        new = [i for i, n in enumerate(names) if " " in n or "^" in n]
        poly_df = pd.DataFrame(poly[:, new], columns=[f"POLY_{names[i]}" for i in new], index=data.index)
        data = pd.concat([data, poly_df.astype(np.float32)], axis=1)

    y = data.pop("TARGET").astype(int)
    X = data.astype(np.float32)
    X.columns = clean_names(X.columns)
    report = {
        "before": before,
        "after": {"rows": len(X), "features": X.shape[1]},
        "dropped_columns_over_threshold": int((missing > threshold).sum()),
        "positive_rate_after": float(y.mean()),
    }
    return X, y, report


def downsample(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED):
    """Keep all minority rows and an equal-sized random subset of the majority."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
    idx = rng.permutation(np.concatenate([pos, neg]))
    return X.iloc[idx], y.iloc[idx]


# ---------------------------------------------------------------------- models
def paper_models(seed: int = config.SEED) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Multi-layer Perceptron": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                          max_iter=300, random_state=seed)),
        "LightGBM": LGBMClassifier(random_state=seed, verbose=-1),
    }


def balancing_comparison(Xtr, ytr, Xte, yte, seed=config.SEED) -> dict:
    """Paper Section V-A: none vs down-sampling vs SMOTE vs class weights."""
    res = {}
    for model_name in ("Logistic Regression", "LightGBM"):
        def base(weight=None):
            if model_name == "LightGBM":
                return LGBMClassifier(random_state=seed, verbose=-1, class_weight=weight)
            return LogisticRegression(max_iter=3000, class_weight=weight)

        setups = {
            "No balancing": (make_pipeline(StandardScaler(), base()), Xtr, ytr),
            "Down-sampling": (make_pipeline(StandardScaler(), base()), *downsample(Xtr, ytr, seed)),
            "SMOTE": (ImbPipeline([("scale", StandardScaler()),
                                   ("smote", SMOTE(random_state=seed)), ("clf", base())]), Xtr, ytr),
            # w_i = n / (2 n_i), exactly the paper's equation (1)
            "Class weights": (make_pipeline(StandardScaler(), base("balanced")), Xtr, ytr),
        }
        for strategy, (model, X, y) in setups.items():
            t0 = time.time()
            model.fit(X, y)
            m = evaluate(yte, model.predict_proba(Xte)[:, 1])
            res[f"{model_name} | {strategy}"] = m
            print(f"  {model_name:20s} {strategy:14s} AUC={m['roc_auc']:.3f} "
                  f"recall(default)={m['recall_1']:.2f} acc={m['accuracy']:.3f} ({time.time()-t0:.0f}s)")
    return res


def kmeans_experiment(Xtr, ytr, Xte_bal, yte_bal, ks=range(2, 7), seed=config.SEED):
    """Cluster the (down-sampled) training set, then fit one LightGBM per cluster."""
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte_bal)
    summary, per_cluster_at_4 = {}, {}

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Ztr)
        ctr, cte = km.labels_, km.predict(Zte)
        proba = np.zeros(len(yte_bal))
        details = {}
        for c in range(k):
            tr_mask, te_mask = ctr == c, cte == c
            if te_mask.sum() == 0:
                continue
            y_c = ytr.to_numpy()[tr_mask]
            if tr_mask.sum() < 30 or y_c.min() == y_c.max():
                proba[te_mask] = y_c.mean() if len(y_c) else 0.5
            else:
                model = LGBMClassifier(random_state=seed, verbose=-1)
                model.fit(Xtr.iloc[np.flatnonzero(tr_mask)], y_c)
                proba[te_mask] = model.predict_proba(Xte_bal.iloc[np.flatnonzero(te_mask)])[:, 1]
            details[f"Cluster {c + 1}"] = evaluate(yte_bal.to_numpy()[te_mask], proba[te_mask])
            details[f"Cluster {c + 1}"]["n_test"] = int(te_mask.sum())
        overall = evaluate(yte_bal, proba)
        summary[k] = overall
        if k == 4:
            per_cluster_at_4 = {**details, "Overall": overall}
        print(f"  k={k}: accuracy={overall['accuracy']:.3f} AUC={overall['roc_auc']:.3f}")
    return summary, per_cluster_at_4


# ---------------------------------------------------------------------- figures
def plot_roc(curves: dict, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color="#9a9893", lw=1, ls="--")
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], label=f"{name} (AUC {auc:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (defaults caught)", title=title,
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8.5)
    savefig(fig, path)


def plot_kmeans(summary: dict, global_lgbm: dict, path):
    ks = sorted(summary)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ks, [summary[k]["accuracy"] for k in ks], marker="o", ms=7, color=COLORS[0],
            label="Per-cluster LightGBM")
    ax.axhline(global_lgbm["accuracy"], color=COLORS[1], ls="--", label="Single LightGBM")
    ax.set(xlabel="Number of clusters k", ylabel="Accuracy (balanced test)",
           title="K-means + per-cluster models", xticks=ks)
    ax.legend(loc="best", fontsize=8.5)
    savefig(fig, path)


# ------------------------------------------------------------------------- main
def run(features_path=None, skip_kmeans=False, seed=config.SEED) -> dict:
    config.ensure_dirs()
    df = load_features(features_path or config.FEATURES_PATH)

    print("[replicate] paper-style cleaning ...")
    X, y, prep = paper_clean(df)
    print(f"  before: {prep['before']}  after: {prep['after']}  "
          f"default rate after cleaning: {prep['positive_rate_after']:.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    Xtr_bal, ytr_bal = downsample(Xtr, ytr, seed)
    Xte_bal, yte_bal = downsample(Xte, yte, seed)
    print(f"  train (down-sampled): {len(Xtr_bal)}  test balanced: {len(Xte_bal)}  test natural: {len(Xte)}")

    print("[replicate] training the paper's five models on down-sampled data ...")
    balanced, natural, curves = {}, {}, {}
    for name, model in paper_models(seed).items():
        t0 = time.time()
        model.fit(Xtr_bal, ytr_bal)
        balanced[name] = evaluate(yte_bal, model.predict_proba(Xte_bal)[:, 1])
        p_nat = model.predict_proba(Xte)[:, 1]
        natural[name] = evaluate(yte, p_nat)
        fpr, tpr, _ = roc_curve(yte, p_nat)
        curves[name] = (fpr, tpr, natural[name]["roc_auc"])
        print(f"  {name:24s} balanced acc={balanced[name]['accuracy']:.3f}  "
              f"AUC={natural[name]['roc_auc']:.3f}  ({time.time() - t0:.0f}s)")

    print("[replicate] balancing strategies (evaluated on natural test) ...")
    balancing = balancing_comparison(Xtr, ytr, Xte, yte, seed)

    kmeans_summary, kmeans_k4 = {}, {}
    if not skip_kmeans:
        print("[replicate] K-means + per-cluster LightGBM ...")
        kmeans_summary, kmeans_k4 = kmeans_experiment(Xtr_bal, ytr_bal, Xte_bal, yte_bal, seed=seed)

    # ---- save
    plot_roc(curves, config.FIGURES_DIR / "replication_roc.png", "Paper models: ROC on natural test set")
    if kmeans_summary:
        plot_kmeans(kmeans_summary, balanced["LightGBM"], config.FIGURES_DIR / "replication_kmeans.png")

    results = {"preprocessing": prep, "balanced_test": balanced, "natural_test": natural,
               "balancing": balancing, "kmeans": {str(k): v for k, v in kmeans_summary.items()},
               "kmeans_k4_per_cluster": kmeans_k4}
    save_json(results, config.RESULTS_DIR / "replication.json")

    md = ["# Experiment 1 - paper replication", "",
          f"Cleaning: {prep['before']['rows']:,} rows x {prep['before']['features']} features -> "
          f"{prep['after']['rows']:,} rows x {prep['after']['features']} features "
          f"({prep['dropped_columns_over_threshold']} columns dropped for >30% missing).", "",
          "## Balanced test set (comparable to paper TABLE III)", to_markdown(paper_style_table(balanced)), "",
          "## Natural (imbalanced) test set", to_markdown(paper_style_table(natural)), "",
          "## Balancing strategies (natural test set)", to_markdown(paper_style_table(balancing)), ""]
    if kmeans_k4:
        md += ["## K-means (k=4) + per-cluster LightGBM (comparable to paper TABLE IV)",
               to_markdown(paper_style_table(kmeans_k4)), "",
               "Accuracy by k: " + ", ".join(f"k={k}: {v['accuracy']:.3f}" for k, v in kmeans_summary.items()), ""]
    (config.RESULTS_DIR / "replication.md").write_text("\n".join(md))
    print(f"[replicate] results -> {config.RESULTS_DIR / 'replication.md'}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=None)
    p.add_argument("--skip-kmeans", action="store_true")
    a = p.parse_args()
    run(a.features, a.skip_kmeans)


if __name__ == "__main__":
    main()
