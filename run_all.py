"""Run every experiment end to end and write results/SUMMARY.md.

    python run_all.py                       # full run on data/
    python run_all.py --sample 60000 --fast # ~quick dev run
    python run_all.py --skip-features       # reuse artifacts/features.pkl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loanliness import config, improve, replicate, visualize
from loanliness.features import build_features

# Paper TABLE III (accuracy on their balanced test set) and TABLE IV overall
PAPER_ACCURACY = {
    "Logistic Regression": 0.6934, "Random Forest": 0.6351, "Naive Bayes": 0.5211,
    "Multi-layer Perceptron": 0.6915, "LightGBM": 0.5747,
}
PAPER_KMEANS_OVERALL = 0.7157


def summary(rep: dict, imp: dict) -> str:
    bal, nat = rep["balanced_test"], rep["natural_test"]
    lines = [
        "# Results summary", "",
        "## Paper vs our replication (accuracy on a balanced test set)", "",
        "| Model | Paper | Ours | Our ROC-AUC (natural test) |", "|---|---|---|---|",
    ]
    for m, acc in PAPER_ACCURACY.items():
        lines.append(f"| {m} | {acc:.2%} | {bal[m]['accuracy']:.2%} | {nat[m]['roc_auc']:.3f} |")
    if rep.get("kmeans"):
        k4 = rep["kmeans"].get("4", {})
        if k4:
            lines.append(f"| K-means (k=4) + per-cluster LightGBM | {PAPER_KMEANS_OVERALL:.2%} | "
                         f"{k4['accuracy']:.2%} | - |")
    best_paper_auc = max(v["roc_auc"] for v in nat.values())
    t = imp["test"]["LightGBM (all rows, all tables)"]
    lines += [
        "", "## Improved pipeline (natural test set, all applicants)", "",
        f"* Rows used: paper-style cleaning keeps {rep['preprocessing']['after']['rows']:,} of "
        f"{rep['preprocessing']['before']['rows']:,}; improved pipeline keeps all {imp['n_rows']:,}.",
        f"* Best paper-style model ROC-AUC: {best_paper_auc:.3f}",
        f"* Improved LightGBM ROC-AUC: **{t['roc_auc']:.3f}** (CV OOF {imp['cv_oof_auc']:.3f}), "
        f"PR-AUC {t['pr_auc']:.3f} vs base rate {imp['default_rate']:.3f}",
        f"* At threshold {imp['threshold']:.2f}: catches {t['recall_1']:.0%} of defaults with "
        f"{t['precision_1']:.0%} precision; balanced accuracy {t['balanced_accuracy']:.1%}",
        "", "## Ablations (ROC-AUC)", "",
        *[f"* {k}: {v:.3f}" for k, v in imp["ablation_auc"].items()],
    ]
    thin = imp.get("thin_file", {})
    if thin.get("auc_full_model"):
        lines += ["", "## Thin-file applicants (no Credit Bureau record)", "",
                  f"* {thin['n']:,} test applicants, default rate {thin['default_rate']:.1%}",
                  f"* ROC-AUC {thin['auc_full_model']:.3f} (vs {thin['auc_with_history']:.3f} with history); "
                  f"without external scores {thin['auc_no_ext_source']:.3f}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--sample", type=int, default=None, help="use N random applicants")
    p.add_argument("--fast", action="store_true", help="faster LightGBM settings, 3 folds")
    p.add_argument("--skip-features", action="store_true", help="reuse artifacts/features.pkl")
    p.add_argument("--skip-kmeans", action="store_true")
    a = p.parse_args()
    config.ensure_dirs()
    t0 = time.time()

    if not a.skip_features:
        build_features(a.data_dir, a.sample).to_pickle(config.FEATURES_PATH)
    rep = replicate.run(skip_kmeans=a.skip_kmeans)
    visualize.run()
    imp = improve.run(fast=a.fast)

    out = config.RESULTS_DIR / "SUMMARY.md"
    out.write_text(summary(rep, imp))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Summary -> {out}")
    print(out.read_text())


if __name__ == "__main__":
    main()
