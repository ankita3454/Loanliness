"""Generate small fake tables with the same schema as the Kaggle Home Credit data.

Only for smoke-testing the pipeline without the 2.5 GB download. The numbers are
random (with a planted signal so models learn something); never report them.

    python tests/make_synthetic_data.py --out data_synthetic --n 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANOMALY = 365243


def make(n: int = 4000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n)

    age_days = -rng.integers(21 * 365, 69 * 365, n)
    pensioner = rng.random(n) < 0.18
    employed = np.where(pensioner, ANOMALY, -rng.integers(0, 40 * 365, n).clip(max=-age_days - 18 * 365))
    income = rng.lognormal(11.9, 0.5, n).round(-2)
    credit = (income * rng.uniform(1, 8, n)).round(-2)
    annuity = (credit * rng.uniform(0.03, 0.09, n)).round(1)
    ext = rng.beta(4, 3, (n, 3))
    ext_obs = ext.copy()
    ext_obs[rng.random(n) < 0.56, 0] = np.nan
    ext_obs[rng.random(n) < 0.002, 1] = np.nan
    ext_obs[rng.random(n) < 0.20, 2] = np.nan
    has_history = rng.random(n) < 0.86
    late_tendency = rng.beta(1.5, 12, n)

    emp_years = np.where(pensioner, 0, -employed / 365)
    logit = (-2.6 - 3.5 * (ext.mean(1) - 0.57) - 0.04 * (emp_years - 6)
             + 0.25 * (credit / income - 4) / 2 + 6 * (late_tendency - 0.11)
             + 0.012 * (age_days / 365 + 43) + rng.normal(0, 0.6, n))
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    pick = lambda opts, p=None: rng.choice(opts, n, p=p)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": pick(["Cash loans", "Revolving loans"], [0.9, 0.1]),
        "CODE_GENDER": pick(["F", "M"], [0.66, 0.34]),
        "FLAG_OWN_CAR": pick(["N", "Y"], [0.66, 0.34]),
        "FLAG_OWN_REALTY": pick(["Y", "N"], [0.69, 0.31]),
        "CNT_CHILDREN": rng.poisson(0.4, n),
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(-2),
        "NAME_TYPE_SUITE": pick(["Unaccompanied", "Family", "Spouse, partner", None], [0.8, 0.13, 0.04, 0.03]),
        "NAME_INCOME_TYPE": np.where(pensioner, "Pensioner",
                                     pick(["Working", "Commercial associate", "State servant"], [0.62, 0.27, 0.11])),
        "NAME_EDUCATION_TYPE": pick(["Secondary / secondary special", "Higher education",
                                     "Incomplete higher", "Lower secondary"], [0.71, 0.24, 0.03, 0.02]),
        "NAME_FAMILY_STATUS": pick(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                                   [0.64, 0.15, 0.10, 0.06, 0.05]),
        "NAME_HOUSING_TYPE": pick(["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
                                  [0.89, 0.05, 0.04, 0.02]),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n).round(6),
        "DAYS_BIRTH": age_days, "DAYS_EMPLOYED": employed,
        "DAYS_REGISTRATION": -rng.integers(0, 20000, n).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
        "OWN_CAR_AGE": np.where(rng.random(n) < 0.66, np.nan, rng.integers(0, 30, n)),
        "OCCUPATION_TYPE": pick(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", None],
                                [0.18, 0.1, 0.09, 0.07, 0.06, 0.5]),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "REGION_RATING_CLIENT": pick([1, 2, 3], [0.1, 0.74, 0.16]),
        "ORGANIZATION_TYPE": pick(["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine"],
                                  [0.3, 0.18, 0.2, 0.2, 0.12]),
        "EXT_SOURCE_1": ext_obs[:, 0], "EXT_SOURCE_2": ext_obs[:, 1], "EXT_SOURCE_3": ext_obs[:, 2],
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
        "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < 0.13, np.nan, rng.poisson(1.9, n)),
    })

    # ---- bureau + bureau_balance (only applicants with history)
    rows, bb_rows, bid = [], [], 5000000
    for i, cid in enumerate(ids):
        if not has_history[i]:
            continue
        for _ in range(rng.integers(1, 7)):
            bid += 1
            amt = round(float(rng.lognormal(12, 1)), -2)
            debt = amt * rng.uniform(0, 1) if rng.random() < 0.4 else 0.0
            overdue = amt * 0.1 if rng.random() < late_tendency[i] else 0.0
            rows.append((cid, bid, rng.choice(["Closed", "Active"], p=[0.63, 0.37]), "currency 1",
                         -int(rng.integers(0, 2900)), int(overdue > 0) * int(rng.integers(1, 60)),
                         float(rng.integers(-2000, 2000)), np.nan if rng.random() < 0.37 else -float(rng.integers(0, 2900)),
                         np.nan if rng.random() < 0.6 else overdue, 0, amt, debt, 0.0, overdue,
                         rng.choice(["Consumer credit", "Credit card", "Car loan"], p=[0.73, 0.23, 0.04]),
                         -int(rng.integers(0, 1500)), np.nan))
            for m in range(int(rng.integers(1, 12))):
                st = "1" if rng.random() < late_tendency[i] else rng.choice(["0", "C", "X"])
                bb_rows.append((bid, -m, st))
    bureau = pd.DataFrame(rows, columns=[
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
        "CREDIT_TYPE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY"])
    bureau_balance = pd.DataFrame(bb_rows, columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"])

    # ---- previous applications + monthly tables
    prev, pos, ins, cc, pid = [], [], [], [], 1000000
    for i, cid in enumerate(ids):
        for _ in range(rng.integers(0, 5)):
            pid += 1
            amt = round(float(rng.lognormal(11.5, 0.8)), -2)
            status = rng.choice(["Approved", "Refused", "Canceled"], p=[0.62, 0.17 + late_tendency[i] / 3, 0.21 - late_tendency[i] / 3])
            nterm = int(rng.choice([6, 12, 24, 36]))
            prev.append((pid, cid, rng.choice(["Cash loans", "Consumer loans", "Revolving loans"]), amt / nterm,
                         amt, amt * rng.uniform(0.9, 1.1), np.nan, amt, status, -int(rng.integers(1, 2900)),
                         nterm, ANOMALY if rng.random() < 0.6 else -float(rng.integers(0, 2900))))
            if status != "Approved":
                continue
            for k in range(min(nterm, 10)):
                pos.append((pid, cid, -k - 1, nterm, max(nterm - k, 0), "Active", 0, 0))
                due = -300 + 30 * k
                lateness = int(rng.integers(1, 40)) if rng.random() < late_tendency[i] else -int(rng.integers(0, 10))
                pay = amt / nterm * (rng.uniform(0.5, 1) if rng.random() < late_tendency[i] else 1)
                ins.append((pid, cid, 1, k + 1, due, due + lateness, amt / nterm, pay))
            if rng.random() < 0.3:
                limit = float(rng.choice([45000, 90000, 135000, 270000]))
                for m in range(6):
                    cc.append((pid, cid, -m - 1, limit * rng.uniform(0, 1.05), limit,
                               float(rng.integers(0, 3)) * 1000, "Active",
                               int(rng.random() < late_tendency[i]) * int(rng.integers(1, 30)), 0))
    previous_application = pd.DataFrame(prev, columns=[
        "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT",
        "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "NAME_CONTRACT_STATUS", "DAYS_DECISION", "CNT_PAYMENT",
        "DAYS_FIRST_DRAWING"])
    pos_cash = pd.DataFrame(pos, columns=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
                                          "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    installments = pd.DataFrame(ins, columns=["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
                                              "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                                              "AMT_INSTALMENT", "AMT_PAYMENT"])
    credit_card = pd.DataFrame(cc, columns=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                                            "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
                                            "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    return {
        "application_train": app, "bureau": bureau, "bureau_balance": bureau_balance,
        "previous_application": previous_application, "POS_CASH_balance": pos_cash,
        "installments_payments": installments, "credit_card_balance": credit_card,
    }


def write(out: Path, n: int = 4000, seed: int = 0) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in make(n, seed).items():
        df.to_csv(out / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data_synthetic"))
    p.add_argument("--n", type=int, default=4000)
    a = p.parse_args()
    print("wrote", write(a.out, a.n))
