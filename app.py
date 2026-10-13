"""Live demo: score a loan applicant and explain the score.

    streamlit run app.py

Needs artifacts/demo_model.joblib (created by `python -m loanliness.improve`).
"""
from __future__ import annotations


import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from loanliness import config
from loanliness.common import COLORS
from loanliness.features import application

st.set_page_config(page_title="Loanliness - repayment risk", page_icon=":bank:", layout="wide")

LABELS = {
    "AMT_INCOME_TOTAL": "Annual income", "AMT_CREDIT": "Loan amount", "AMT_ANNUITY": "Yearly repayment",
    "AMT_GOODS_PRICE": "Price of goods", "DAYS_BIRTH": "Age", "DAYS_EMPLOYED": "Time employed",
    "DAYS_EMPLOYED_ANOMALY": "Pensioner / not employed", "CNT_CHILDREN": "Children",
    "CNT_FAM_MEMBERS": "Family size", "NAME_CONTRACT_TYPE": "Loan type", "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education", "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing", "OCCUPATION_TYPE": "Occupation", "FLAG_OWN_CAR": "Owns car",
    "FLAG_OWN_REALTY": "Owns property", "REGION_RATING_CLIENT": "Region rating",
    "EXT_SOURCE_1": "External score 1", "EXT_SOURCE_2": "External score 2", "EXT_SOURCE_3": "External score 3",
    "EXT_SOURCE_MEAN": "Mean external score", "EXT_SOURCE_STD": "Spread of external scores",
    "CREDIT_INCOME_RATIO": "Loan / income", "ANNUITY_INCOME_RATIO": "Repayment / income",
    "PAYMENT_RATE": "Repayment / loan", "GOODS_CREDIT_RATIO": "Goods price / loan",
    "EMPLOYED_AGE_RATIO": "Share of life employed", "INCOME_PER_PERSON": "Income per family member",
}


@st.cache_resource
def load_bundle():
    if not config.DEMO_MODEL_PATH.exists():
        return None
    return joblib.load(config.DEMO_MODEL_PATH)


def options(bundle, col, fallback):
    return [c for c in bundle["categories"].get(col, fallback) if c not in ("OTHER",)] or fallback


def predict(bundle, raw: dict):
    row = application(pd.DataFrame([raw]))
    X = pd.DataFrame({c: row[c] if c in row else np.nan for c in bundle["features"]})
    for c, levels in bundle["categories"].items():
        X[c] = pd.Categorical(X[c], categories=levels)
    model = bundle["model"]
    proba = float(model.predict_proba(X)[:, 1][0])
    contrib = model.booster_.predict(X, pred_contrib=True)[0]
    contrib = pd.Series(contrib[:-1], index=bundle["features"])
    return proba, contrib


def contribution_chart(contrib: pd.Series, top=8):
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [COLORS[7] if v > 0 else COLORS[0] for v in s.values]
    ax.barh([LABELS.get(i, i) for i in s.index], s.values, color=colors, height=0.6)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("Effect on risk (log-odds)   <- lowers risk  |  raises risk ->", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def scorer(bundle):
    st.subheader("Applicant")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 21, 69, 35)
        pensioner = st.checkbox("Pensioner / not currently employed", value=False)
        years_emp = st.slider("Years employed", 0.0, 40.0, 5.0, 0.5, disabled=pensioner)
        income = st.number_input("Annual income", 25_000, 5_000_000, 180_000, step=5_000)
        children = st.number_input("Children", 0, 10, 0)
        family = st.number_input("Family members", 1, 12, 2)
    with c2:
        credit = st.number_input("Loan amount", 45_000, 4_000_000, 600_000, step=10_000)
        annuity = st.number_input("Yearly repayment (annuity)", 1_000, 300_000, 30_000, step=1_000)
        goods = st.number_input("Price of goods bought", 0, 4_000_000, 540_000, step=10_000)
        contract = st.selectbox("Loan type", options(bundle, "NAME_CONTRACT_TYPE", ["Cash loans"]))
        region = st.select_slider("Region rating (1 = best)", [1, 2, 3], 2)
    with c3:
        income_type = st.selectbox("Income type", options(bundle, "NAME_INCOME_TYPE", ["Working"]))
        education = st.selectbox("Education", options(bundle, "NAME_EDUCATION_TYPE", ["Higher education"]))
        fam_status = st.selectbox("Family status", options(bundle, "NAME_FAMILY_STATUS", ["Married"]))
        housing = st.selectbox("Housing", options(bundle, "NAME_HOUSING_TYPE", ["House / apartment"]))
        occupation = st.selectbox("Occupation", ["(not given)"] + options(bundle, "OCCUPATION_TYPE", []))
        own_car = st.toggle("Owns a car", False)
        own_realty = st.toggle("Owns property", True)

    st.markdown("**External credit scores** (normalised 0-1, from credit bureaus). "
                "Untick to simulate a *thin-file* applicant, the group the paper is about.")
    e = st.columns(3)
    ext = []
    for i, col in enumerate(e):
        with col:
            have = st.checkbox(f"Score {i + 1} available", value=i > 0, key=f"have{i}")
            v = st.slider(f"External score {i + 1}", 0.0, 1.0, 0.5, 0.01, disabled=not have, key=f"ext{i}")
            ext.append(v if have else np.nan)

    raw = {
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": -age * 365, "DAYS_EMPLOYED": config.DAYS_ANOMALY if pensioner else -int(years_emp * 365),
        "CNT_CHILDREN": children, "CNT_FAM_MEMBERS": family, "NAME_CONTRACT_TYPE": contract,
        "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": fam_status,
        "NAME_HOUSING_TYPE": housing, "OCCUPATION_TYPE": None if occupation == "(not given)" else occupation,
        "FLAG_OWN_CAR": "Y" if own_car else "N", "FLAG_OWN_REALTY": "Y" if own_realty else "N",
        "REGION_RATING_CLIENT": region,
        "EXT_SOURCE_1": ext[0], "EXT_SOURCE_2": ext[1], "EXT_SOURCE_3": ext[2],
    }
    proba, contrib = predict(bundle, raw)
    thr, base = bundle["threshold"], bundle["base_rate"]

    st.divider()
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Prediction")
        st.metric("Probability of payment difficulty", f"{proba:.1%}",
                  delta=f"{(proba - base) * 100:+.1f} pts vs average applicant ({base:.1%})",
                  delta_color="inverse")
        if proba >= thr:
            st.error(f"High risk - above the decision threshold ({thr:.0%}). Flag for manual review.")
        else:
            st.success(f"Lower risk - below the decision threshold ({thr:.0%}).")
        st.caption(f"Demo model: LightGBM on {len(bundle['features'])} application fields, "
                   f"test ROC-AUC {bundle['test_auc']:.3f}. Gender is deliberately not used.")
    with right:
        st.subheader("Why? Top factors for this applicant")
        st.pyplot(contribution_chart(contrib))


def results_tab():
    st.subheader("Experiment results")
    for name in ("replication.md", "improved.md"):
        p = config.RESULTS_DIR / name
        if p.exists():
            with st.expander(name, expanded=name == "improved.md"):
                st.markdown(p.read_text())
    figs = sorted(config.FIGURES_DIR.glob("*.png"))
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem.replace("_", " "))
    if not figs:
        st.info("No figures yet - run `python run_all.py` first.")


def main():
    st.title("Loanliness - predicting loan repayment ability")
    bundle = load_bundle()
    tab1, tab2 = st.tabs(["Score an applicant", "Results"])
    with tab1:
        if bundle is None:
            st.warning("No trained model found. Run `python -m loanliness.improve` (or `python run_all.py`) first.")
        else:
            scorer(bundle)
    with tab2:
        results_tab()


main()
