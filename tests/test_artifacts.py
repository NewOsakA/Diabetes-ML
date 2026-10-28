import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.common import ROOT, FEATURES, evaluate

RAW_CSV = ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"


@pytest.mark.skipif(not (ROOT / "data/processed/splits.npz").exists() or not RAW_CSV.exists(), reason="Download the dataset and run training if saved artifacts are absent")
def test_splits_cover_every_row_without_overlap_and_source_matches():
    splits = np.load(ROOT / "data/processed/splits.npz")
    joined = np.concatenate([splits[name] for name in splits.files])
    np.testing.assert_array_equal(np.sort(joined), np.arange(253680))
    assert len(np.unique(joined)) == len(joined)
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    path = ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["source_sha256"]


@pytest.mark.skipif(not (ROOT / "results/test_results.csv").exists() or not RAW_CSV.exists(), reason="Download the dataset and run training if saved artifacts are absent")
def test_frozen_selection_and_saved_inference_match_recorded_predictions():
    from diabetes_ml.predict import Predictor
    manifest = json.loads((ROOT / "models/selection_manifest.json").read_text())
    experiments = pd.read_csv(ROOT / "results/experiment_results.csv")
    selected = experiments[experiments.split == "selection"].sort_values(["macro_f1", "balanced_accuracy"], ascending=False, kind="stable").iloc[0]
    assert manifest["champion"]["model"] == selected.model
    assert manifest["champion"]["decision"] == selected.decision
    assert manifest["test_used_for_selection"] is False
    raw = pd.read_csv(ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv")
    expected = pd.read_csv(ROOT / "results/champion_test_predictions.csv")
    sample = expected.sample(n=128, random_state=1)
    predictor = Predictor()
    actual = predictor.predict(raw.iloc[sample.row_id])
    np.testing.assert_array_equal(actual.predicted_class, sample.predicted_class)
    for c in range(3):
        np.testing.assert_allclose(actual[f"raw_score_class_{c}"], sample[f"raw_score_class_{c}"], atol=1e-6)
    with pytest.raises(ValueError, match="Missing features"):
        predictor.predict(raw.drop(columns=[FEATURES[0]]).head())
    invalid = raw.head().copy()
    invalid.loc[0, "Age"] = 72  # Years are not category codes.
    with pytest.raises(ValueError, match="Age"):
        predictor.predict(invalid)
