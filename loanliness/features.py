"""Join the 7 Home Credit tables into one row per applicant.

Paper, Section IV-A ("Feature concatenation"): every auxiliary table is linked to
application_train through SK_ID_CURR (bureau_balance goes via SK_ID_BUREAU first).
Auxiliary tables have many rows per applicant, so each one is aggregated
(mean/max/min for numbers, share-of-rows for categories) before the join.

Usage:
    python -m loanliness.features --data-dir data
    python -m loanliness.features --data-dir data --sample 50000   # faster dev run
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

MAX_CATEGORY_LEVELS = 20


# --------------------------------------------------------------------------- io
def clean_names(cols) -> list[str]:
    """LightGBM rejects JSON-special characters; make names [A-Za-z0-9_] only."""
    out = []
    for c in cols:
        n = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        out.append(re.sub(r"_+", "_", n))
    return out


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Kaggle 'home-credit-default-risk' data "
            f"and unzip it into {data_dir}/ (see README)."
        )
    df = pd.read_csv(path)
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype(np.float32)
    return df


# ---------------------------------------------------------------------- helpers
def _cap_levels(s: pd.Series, k: int = MAX_CATEGORY_LEVELS) -> pd.Series:
    top = s.value_counts().index[:k]
    return s.where(s.isin(top) | s.isna(), "OTHER")


def _fix_days(df: pd.DataFrame) -> pd.DataFrame:
    days = [c for c in df.columns if c.startswith("DAYS_")]
    df[days] = df[days].replace(config.DAYS_ANOMALY, np.nan)
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def aggregate(df: pd.DataFrame, key: str, prefix: str,
              num_aggs=("mean", "max", "min"), drop=()) -> pd.DataFrame:
    """Aggregate a many-rows-per-key table to one row per key."""
    df = df.drop(columns=[c for c in drop if c in df.columns])
    cats = [c for c in df.columns if c != key and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c != key and c not in cats]
    parts = [df.groupby(key).size().rename(f"{prefix}_COUNT").astype(np.float32)]

    if nums:
        g = df.groupby(key)[nums].agg(list(num_aggs))
        g.columns = [f"{prefix}_{c}_{a.upper()}" for c, a in g.columns]
        parts.append(g.astype(np.float32))
    if cats:
        capped = df[cats].apply(_cap_levels)
        d = pd.get_dummies(capped, dtype=np.float32)
        d[key] = df[key].to_numpy()
        g = d.groupby(key).mean()
        g.columns = [f"{prefix}_{c}_SHARE" for c in g.columns]
        parts.append(g)
    return pd.concat(parts, axis=1)


# ----------------------------------------------------------------- per table
def application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == config.DAYS_ANOMALY).astype(np.float32)
    df = _fix_days(df)
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["PAYMENT_RATE"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["EMPLOYED_AGE_RATIO"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    if "CNT_FAM_MEMBERS" in df:
        df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df]
    if ext:
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1).astype(np.float32)
        df["EXT_SOURCE_STD"] = df[ext].std(axis=1).astype(np.float32)
    return df


def bureau(bur: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    bb_agg = aggregate(bb, "SK_ID_BUREAU", "BB", num_aggs=("min", "max"))
    bur = _fix_days(bur.copy()).join(bb_agg, on="SK_ID_BUREAU")
    bur["DEBT_CREDIT_RATIO"] = _safe_div(bur["AMT_CREDIT_SUM_DEBT"], bur["AMT_CREDIT_SUM"])
    bur["OVERDUE_DEBT_RATIO"] = _safe_div(bur["AMT_CREDIT_SUM_OVERDUE"], bur["AMT_CREDIT_SUM_DEBT"])
    return aggregate(bur, "SK_ID_CURR", "BURO", drop=["SK_ID_BUREAU"])


def previous(prev: pd.DataFrame) -> pd.DataFrame:
    prev = _fix_days(prev.copy())
    prev["APP_CREDIT_RATIO"] = _safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    return aggregate(prev, "SK_ID_CURR", "PREV", drop=["SK_ID_PREV"])


def pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    return aggregate(pos, "SK_ID_CURR", "POS", drop=["SK_ID_PREV"])


def installments(ins: pd.DataFrame) -> pd.DataFrame:
    ins = ins.copy()
    late = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DAYS_PAST_DUE"] = late.clip(lower=0).astype(np.float32)
    ins["DAYS_BEFORE_DUE"] = (-late).clip(lower=0).astype(np.float32)
    ins["PAID_LATE"] = (late > 0).astype(np.float32)
    ins["PAYMENT_SHORTFALL"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype(np.float32)
    ins["PAYMENT_RATIO"] = _safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    return aggregate(ins, "SK_ID_CURR", "INSTAL", drop=["SK_ID_PREV"])


def credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["UTILIZATION"] = _safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    return aggregate(cc, "SK_ID_CURR", "CC", drop=["SK_ID_PREV"])


# ------------------------------------------------------------------- pipeline
def build_features(data_dir: Path, sample: int | None = None, seed: int = config.SEED,
                   verbose: bool = True) -> pd.DataFrame:
    def log(msg):
        if verbose:
            print(f"[features] {msg}", flush=True)

    t0 = time.time()
    app = read_table(data_dir, "application_train")
    log(f"application_train: {app.shape}")
    if sample and sample < len(app):
        app = app.sample(sample, random_state=seed)
        log(f"sampled {sample} applicants")
    ids = set(app["SK_ID_CURR"])

    def load(name):
        df = read_table(data_dir, name)
        if "SK_ID_CURR" in df:
            df = df[df["SK_ID_CURR"].isin(ids)]
        log(f"{name}: {df.shape}")
        return df

    df = application(app).set_index("SK_ID_CURR").copy()

    bur = load("bureau")
    bb = read_table(data_dir, "bureau_balance")
    bb = bb[bb["SK_ID_BUREAU"].isin(set(bur["SK_ID_BUREAU"]))]
    log(f"bureau_balance: {bb.shape}")
    blocks = [
        bureau(bur, bb),
        previous(load("previous_application")),
        pos_cash(load("POS_CASH_balance")),
        installments(load("installments_payments")),
        credit_card(load("credit_card_balance")),
    ]
    df = pd.concat([df.copy(), *[b.reindex(df.index) for b in blocks]], axis=1).copy()

    num = df.select_dtypes("number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = pd.concat([df.index.to_frame(index=False), df.reset_index(drop=True)], axis=1)
    df.columns = clean_names(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    log(f"final feature table: {df.shape[0]} rows x {df.shape[1]} columns "
        f"({time.time() - t0:.0f}s)")
    return df


def load_features(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m loanliness.features` first.")
    return pd.read_pickle(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if not pd.api.types.is_numeric_dtype(df[c])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--out", type=Path, default=config.FEATURES_PATH)
    p.add_argument("--sample", type=int, default=None, help="use N random applicants (dev runs)")
    args = p.parse_args()
    config.ensure_dirs()
    df = build_features(args.data_dir, args.sample)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"[features] saved -> {args.out}")


if __name__ == "__main__":
    main()
