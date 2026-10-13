"""Smoke tests on synthetic data with the Kaggle schema.   Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loanliness import config, improve, replicate  # noqa: E402
from loanliness.features import build_features  # noqa: E402
import make_synthetic_data  # noqa: E402


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("loanliness")
    data = make_synthetic_data.write(root / "data", n=2500, seed=1)
    mp = pytest.MonkeyPatch()
    for name, sub in [("ARTIFACTS_DIR", "artifacts"), ("RESULTS_DIR", "results"),
                      ("FIGURES_DIR", "results/figures")]:
        mp.setattr(config, name, root / sub)
    mp.setattr(config, "FEATURES_PATH", root / "artifacts/features.pkl")
    mp.setattr(config, "DEMO_MODEL_PATH", root / "artifacts/demo_model.joblib")
    config.ensure_dirs()
    df = build_features(data, verbose=False)
    df.to_pickle(config.FEATURES_PATH)
    yield root, df
    mp.undo()


def test_one_row_per_applicant(workspace):
    _, df = workspace
    assert df["SK_ID_CURR"].is_unique
    assert len(df) == 2500
    assert not np.isinf(df.select_dtypes("number").to_numpy()).any()
    assert all(c.replace("_", "").isalnum() for c in df.columns), "names must be LightGBM-safe"


def test_days_anomaly_removed(workspace):
    _, df = workspace
    assert (df["DAYS_EMPLOYED"] != config.DAYS_ANOMALY).all()
    assert df["DAYS_EMPLOYED_ANOMALY"].sum() > 0


def test_paper_cleaning_has_no_missing(workspace):
    _, df = workspace
    X, y, report = replicate.paper_clean(df)
    assert not X.isna().any().any()
    assert report["after"]["rows"] < report["before"]["rows"]
    assert set(y.unique()) == {0, 1}


def test_downsample_is_balanced(workspace):
    _, df = workspace
    X, y, _ = replicate.paper_clean(df)
    _, yb = replicate.downsample(X, y)
    assert yb.mean() == pytest.approx(0.5)


def test_improve_runs_and_saves_demo_model(workspace):
    root, _ = workspace
    res = improve.run(fast=True)
    assert 0.5 <= res["test"]["LightGBM (all rows, all tables)"]["roc_auc"] <= 1.0
    assert config.DEMO_MODEL_PATH.exists()
    assert (root / "results/improved.md").exists()
