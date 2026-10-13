"""Paths and constants shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FEATURES_PATH = ARTIFACTS_DIR / "features.pkl"
DEMO_MODEL_PATH = ARTIFACTS_DIR / "demo_model.joblib"

SEED = 42
TEST_SIZE = 0.20

# Kaggle encodes "not applicable" day counts (e.g. pensioners' DAYS_EMPLOYED) as 365243.
DAYS_ANOMALY = 365243

TABLES = [
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
]

# Paper settings (Section IV-A)
PAPER_MISSING_THRESHOLD = 0.30
PAPER_POLY_COLUMNS = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
]


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
