"""Experiment 2: fix the paper's weak spots.

What changes versus the paper:
  * keep ALL applicants (no row dropping) - LightGBM handles NaN natively
  * keep ALL engineered features from the 7 tables (no 30%-missing column cut)
  * evaluate on the natural class balance with ROC-AUC / PR-AUC (Kaggle's metric)
  * 5-fold stratified CV with early stopping; decision threshold tuned on
    out-of-fold predictions only (never on the test set)
  * ablations: without EXT_SOURCE_* (external credit scores) and application-table
    only - this tests the paper's own motivation (people without credit history)
  * a compact "demo" model on fields a person can type in, for the Streamlit app

Usage:  python -m loanliness.improve            # full run
        python -m loanliness.improve --fast     # quicker, for development
"""
from __future__ import annotations

import argparse
import inspect
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .common import (COLORS, best_f1_threshold, evaluate, paper_style_table, plt, save_json,
                     savefig, to_markdown)
from .features import categorical_columns, feature_columns, load_features

HISTORY_PREFIXES = ("BURO_", "PREV_", "POS_", "INSTAL_", "CC_")

# Fields a loan officer could type in (CODE_GENDER deliberately excluded).
DEMO_RAW_FIELDS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
DEMO_DERIVED = ["DAYS_EMPLOYED_ANOMALY", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "PAYMENT_RATE",
                "GOODS_CREDIT_RATIO", "EMPLOYED_AGE_RATIO", "INCOME_PER_PERSON",
                "EXT_SOURCE_MEAN", "EXT_SOURCE_STD"]


def lgb_params(fast: bool, seed: int) -> dict:
    return dict(n_estimators=5000, learning_rate=0.08 if fast else 0.03, num_leaves=31,
                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)


def _fit(X, y, Xval, yval, params):
    model = lgb.LGBMClassifier(**params)
    kw = dict(eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    if "eval_X" in inspect.signature(model.fit).parameters:   # lightgbm >= 4.7
        model.fit(X, y, eval_X=(Xval,), eval_y=(yval,), **kw)
    else:
        model.fit(X, y, eval_set=[(Xval, yval)], **kw)
    return model


def prepare(df: pd.DataFrame):
    X = df[feature_columns(df)].copy()
    for c in categorical_columns(df):
        X[c] = X[c].astype("category")
    return X, df["TARGET"].astype(int)


def cross_validate(Xtr, ytr, Xte, params, folds, seed):
    oof = np.zeros(len(Xtr))
    test_pred = np.zeros(len(Xte))
    gain = pd.Series(0.0, index=Xtr.columns)
    iters = []
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for k, (a, b) in enumerate(skf.split(Xtr, ytr), 1):
        t0 = time.time()
        m = _fit(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], ytr.iloc[b], params)
        oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
        test_pred += m.predict_proba(Xte)[:, 1] / folds
        gain += pd.Series(m.booster_.feature_importance("gain"), index=Xtr.columns) / folds
        iters.append(m.best_iteration_)
        print(f"  fold {k}/{folds}: AUC={roc_auc_score(ytr.iloc[b], oof[b]):.4f} "
              f"trees={m.best_iteration_} ({time.time() - t0:.0f}s)")
    return oof, test_pred, gain, iters


def single_split_auc(Xtr, ytr, Xte, yte, cols, params, seed):
    a, b = train_test_split(np.arange(len(Xtr)), test_size=0.15, stratify=ytr, random_state=seed)
    m = _fit(Xtr.iloc[a][cols], ytr.iloc[a], Xtr.iloc[b][cols], ytr.iloc[b], params)
    return m, m.predict_proba(Xte[cols])[:, 1]


def logistic_baseline(Xtr, ytr, Xte):
    cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in Xtr.columns if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=0.005)), cats),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
    to_obj = lambda X: X.astype({c: object for c in cats})
    model.fit(to_obj(Xtr), ytr)
    return model.predict_proba(to_obj(Xte))[:, 1]


# ---------------------------------------------------------------------- figures
def plot_curves(yte, preds: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot([0, 1], [0, 1], color="#9a9893", lw=1, ls="--")
    for i, (name, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(yte, p)
        axes[0].plot(fpr, tpr, color=COLORS[i], label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        prec, rec, _ = precision_recall_curve(yte, p)
        axes[1].plot(rec, prec, color=COLORS[i], label=name)
    axes[1].axhline(yte.mean(), color="#9a9893", lw=1, ls="--", label=f"Base rate ({yte.mean():.1%})")
    axes[0].set(title="ROC - natural test set", xlabel="False positive rate",
                ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(title="Precision-recall - default class", xlabel="Recall (defaults caught)",
                ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].legend(loc="upper right", fontsize=8.5)
    savefig(fig, path)


def plot_importance(gain: pd.Series, path, top=25):
    imp = (gain / gain.sum()).sort_values(ascending=False).head(top)
    colors = [COLORS[1] if c.startswith(HISTORY_PREFIXES) else COLORS[0] for c in imp.index[::-1]]
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors, height=0.7)
    ax.set(title=f"LightGBM feature importance (top {top}, share of gain)", xlabel="Share of total gain")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[0], label="Application table"),
                       Patch(color=COLORS[1], label="Credit-history tables")], loc="lower right", fontsize=8.5)
    savefig(fig, path)


def plot_ablation(rows: dict, path):
    names = list(rows)
    vals = [rows[n] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(names[::-1], vals[::-1], color=COLORS[0], height=0.6)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    lo = max(0.5, min(vals) - 0.05)
    ax.set(xlim=(lo, min(1.0, max(vals) + 0.04)), xlabel="ROC-AUC (natural test set)",
           title="Ablation: what drives performance?")
    ax.grid(axis="y", visible=False)
    savefig(fig, path)


# ------------------------------------------------------------------------- demo
def train_demo_model(Xtr, ytr, Xte, yte, params, seed) -> dict:
    cols = [c for c in DEMO_RAW_FIELDS + DEMO_DERIVED if c in Xtr.columns]
    model, p = single_split_auc(Xtr, ytr, Xte, yte, cols, params, seed)
    cats = {c: list(Xtr[c].cat.categories) for c in cols if isinstance(Xtr[c].dtype, pd.CategoricalDtype)}
    num_ranges = {c: [float(Xtr[c].quantile(0.01)), float(Xtr[c].median()), float(Xtr[c].quantile(0.99))]
                  for c in cols if c not in cats}
    bundle = {"model": model, "features": cols, "categories": cats, "num_ranges": num_ranges,
              "threshold": None,  # set by run() from out-of-fold predictions
              "test_auc": float(roc_auc_score(yte, p)), "base_rate": float(ytr.mean())}
    return bundle, p


# ------------------------------------------------------------------------- main
def run(features_path=None, fast=False, folds=5, seed=config.SEED) -> dict:
    config.ensure_dirs()
    df = load_features(features_path or config.FEATURES_PATH)
    X, y = prepare(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    params = lgb_params(fast, seed)
    folds = 3 if fast else folds
    print(f"[improve] {len(X):,} applicants x {X.shape[1]} features "
          f"(default rate {y.mean():.2%}); train {len(Xtr):,} / test {len(Xte):,}")

    print(f"[improve] LightGBM, {folds}-fold CV on all rows, NaN kept ...")
    oof, p_lgb, gain, iters = cross_validate(Xtr, ytr, Xte, params, folds, seed)
    cv_auc = roc_auc_score(ytr, oof)
    thr = best_f1_threshold(ytr.to_numpy(), oof)
    print(f"  OOF AUC={cv_auc:.4f}; threshold (max F1 on OOF)={thr:.2f}")

    print("[improve] logistic-regression baseline on the same split ...")
    p_lr = logistic_baseline(Xtr, ytr, Xte)

    print("[improve] ablations ...")
    all_cols = list(X.columns)
    no_ext = [c for c in all_cols if not c.startswith("EXT_SOURCE")]
    app_only = [c for c in all_cols if not c.startswith(HISTORY_PREFIXES)]
    app_no_ext = [c for c in app_only if not c.startswith("EXT_SOURCE")]
    ablation_preds = {}
    for name, cols in [("No EXT_SOURCE scores", no_ext), ("Application table only", app_only),
                       ("Application only, no EXT_SOURCE", app_no_ext)]:
        _, ablation_preds[name] = single_split_auc(Xtr, ytr, Xte, yte, cols, params, seed)
        print(f"  {name:34s} AUC={roc_auc_score(yte, ablation_preds[name]):.4f}")

    # ---- thin-file applicants: no Credit Bureau record at all
    thin = Xte["BURO_COUNT"].isna().to_numpy() if "BURO_COUNT" in Xte else np.zeros(len(Xte), bool)
    thin_res = {"n": int(thin.sum()), "default_rate": float(yte[thin].mean()) if thin.any() else None}
    if thin.sum() > 20 and yte[thin].nunique() == 2:
        thin_res["auc_full_model"] = float(roc_auc_score(yte[thin], p_lgb[thin]))
        thin_res["auc_no_ext_source"] = float(roc_auc_score(yte[thin], ablation_preds["No EXT_SOURCE scores"][thin]))
        thin_res["auc_with_history"] = float(roc_auc_score(yte[~thin], p_lgb[~thin]))
        print(f"  thin-file test applicants: {thin_res['n']} -> AUC {thin_res['auc_full_model']:.4f} "
              f"(with history: {thin_res['auc_with_history']:.4f})")

    print("[improve] demo model for the Streamlit app ...")
    demo, p_demo = train_demo_model(Xtr, ytr, Xte, yte, params, seed)
    demo["threshold"] = thr
    joblib.dump(demo, config.DEMO_MODEL_PATH)
    print(f"  demo model ({len(demo['features'])} fields) test AUC={demo['test_auc']:.4f} -> {config.DEMO_MODEL_PATH}")

    # ---- metrics
    main_rows = {
        "LightGBM (all rows, all tables)": evaluate(yte, p_lgb, thr),
        "Logistic Regression (same data)": evaluate(yte, p_lr, 0.5),  # class_weight="balanced" centres it at 0.5
        "Demo model (20 typed-in fields)": evaluate(yte, p_demo, thr),
    }
    ablation_auc = {"Full model": main_rows["LightGBM (all rows, all tables)"]["roc_auc"],
                    **{k: float(roc_auc_score(yte, v)) for k, v in ablation_preds.items()}}

    plot_curves(yte, {"LightGBM": p_lgb, "Logistic Regression": p_lr,
                      "LightGBM, no EXT_SOURCE": ablation_preds["No EXT_SOURCE scores"]},
                config.FIGURES_DIR / "improved_roc_pr.png")
    plot_importance(gain, config.FIGURES_DIR / "improved_feature_importance.png")
    plot_ablation(ablation_auc, config.FIGURES_DIR / "ablation_auc.png")

    results = {
        "n_rows": len(X), "n_features": X.shape[1], "default_rate": float(y.mean()),
        "cv_oof_auc": float(cv_auc), "cv_best_iterations": iters, "threshold": thr,
        "test": main_rows, "ablation_auc": ablation_auc, "thin_file": thin_res,
        "top_features": (gain / gain.sum()).sort_values(ascending=False).head(25).round(5).to_dict(),
    }
    save_json(results, config.RESULTS_DIR / "improved.json")
    md = ["# Experiment 2 - improved pipeline", "",
          f"{len(X):,} applicants x {X.shape[1]} features, default rate {y.mean():.2%}. "
          f"{folds}-fold CV OOF ROC-AUC: **{cv_auc:.4f}**. Threshold {thr:.2f} picked on OOF predictions.", "",
          "## Held-out test set (natural class balance)", to_markdown(paper_style_table(main_rows)), "",
          "## Ablations (ROC-AUC)", "| Setting | ROC-AUC |", "|---|---|",
          *[f"| {k} | {v:.4f} |" for k, v in ablation_auc.items()], "",
          "## Thin-file applicants (no Credit Bureau record)",
          "```", str(thin_res), "```", ""]
    (config.RESULTS_DIR / "improved.md").write_text("\n".join(md))
    print(f"[improve] results -> {config.RESULTS_DIR / 'improved.md'}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=None)
    p.add_argument("--fast", action="store_true", help="higher learning rate, 3 folds")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    run(a.features, a.fast, a.folds)


if __name__ == "__main__":
    main()
