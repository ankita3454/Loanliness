"""Paper Section V-D/E: PCA, t-SNE and random-forest feature importance.

Usage:  python -m loanliness.visualize [--n 5000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from . import config
from .common import DEFAULT, REPAID, plt, save_json, savefig
from .features import load_features
from .replicate import downsample, paper_clean


def _scatter(ax, Z, y, title):
    for label, color, name in ((0, REPAID, "Repaid (0)"), (1, DEFAULT, "Default (1)")):
        m = y == label
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.45, color=color, label=name, linewidths=0)
    ax.set(title=title, xticks=[], yticks=[])
    ax.grid(False)


def run(features_path=None, n=5000, seed=config.SEED) -> dict:
    config.ensure_dirs()
    X, y, _ = paper_clean(load_features(features_path or config.FEATURES_PATH))
    Xb, yb = downsample(X, y, seed)
    if len(Xb) > n:
        Xb, yb = Xb.iloc[:n], yb.iloc[:n]
    Z = StandardScaler().fit_transform(Xb)
    yv = yb.to_numpy()
    print(f"[visualize] {len(Xb)} balanced points, {Z.shape[1]} features")

    pca = PCA(n_components=min(50, Z.shape[1]), random_state=seed).fit(Z)
    Zp = pca.transform(Z)
    var2 = float(pca.explained_variance_ratio_[:2].sum())
    perplexity = min(30, max(5, len(Z) // 10))
    Zt = TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Z)
    Zpt = TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Zp)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _scatter(axes[0], Zp[:, :2], yv, f"PCA (top-2 PCs explain {var2:.0%})")
    _scatter(axes[1], Zt, yv, "t-SNE")
    _scatter(axes[2], Zpt, yv, "t-SNE after PCA-50")
    axes[0].legend(loc="upper right", markerscale=3, fontsize=8.5)
    savefig(fig, config.FIGURES_DIR / "pca_tsne.png")

    # Random-forest importance (paper Fig. 3)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(X.loc[Xb.index], yb)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(imp.index[::-1], imp.values[::-1], color=REPAID, height=0.7)
    ax.set(title="Random forest feature importance (top 20)", xlabel="Mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, config.FIGURES_DIR / "rf_feature_importance.png")

    out = {"pca_top2_variance": var2, "rf_top20": imp.round(5).to_dict()}
    save_json(out, config.RESULTS_DIR / "visualize.json")
    print(f"[visualize] top-2 PCs explain {var2:.1%} of variance; top RF feature: {imp.index[0]}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=None)
    p.add_argument("--n", type=int, default=5000, help="points to embed (t-SNE is slow)")
    a = p.parse_args()
    run(a.features, a.n)


if __name__ == "__main__":
    main()
