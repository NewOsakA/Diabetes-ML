import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from diabetes_ml.common import FEATURES
from diabetes_ml.v2_protocol import partition_roles, validate_roles, tune_decisions
from diabetes_ml.v2_models import VectorCalibrator, resample_training, fit_model, save_bundle, load_bundle, predict_raw
from diabetes_ml.v2_run import select_candidate


def test_nested_roles_are_disjoint_and_exclude_historical_test():
    y = np.array([0]*300+[1]*60+[2]*90)
    roles = partition_roles(np.arange(420), y, seed=5)
    validate_roles(roles, np.arange(420, 450), np.arange(450), np.arange(450, 500))
    assert len(np.concatenate(list(roles.values()))) == 420
    for ids in roles.values():
        assert set(y[ids]) == {0, 1, 2}
    with pytest.raises(ValueError, match="disjoint"):
        validate_roles(roles, roles["fit"][:2], np.arange(450), np.arange(450, 500))
    with pytest.raises(ValueError, match="Historical"):
        validate_roles(roles, np.arange(450, 455), np.arange(450), np.arange(450, 500))


def test_calibration_normalizes_and_corrects_known_prior_shift():
    y = np.array([0]*700+[1]*100+[2]*200)
    p = np.tile([.3, .4, .3], (len(y), 1))
    cal = VectorCalibrator().fit(p, y)
    corrected = cal.transform(p)
    np.testing.assert_allclose(corrected.sum(1), 1.)
    assert log_loss(y, corrected) < log_loss(y, p)
    np.testing.assert_allclose(corrected.mean(0), [.7, .1, .2], atol=.015)


def test_recall_constraint_reports_infeasibility_instead_of_claiming_success():
    y = np.array([0]*100+[1]*2+[2]*10)
    p = np.tile([.8, .02, .18], (len(y), 1))
    rules = tune_decisions(y, p, floor=.9, min_predictions=30)
    assert not rules["recall_constraint_feasible"]
    assert rules["recall"] == rules["macro"]


def test_candidate_selection_averages_inner_folds_and_observes_precision_floor():
    table = pd.DataFrame([
        {"name": "a", "kind": "mlp", "mode": "raw_macro", "macro_f1": v,
         "balanced_accuracy": .5, "prediabetes_precision": .03, "prediabetes_recall": .8}
        for v in [.7, .2]
    ] + [
        {"name": "b", "kind": "mlp", "mode": "calibrated_recall", "macro_f1": .48,
         "balanced_accuracy": .5, "prediabetes_precision": .06, "prediabetes_recall": .2}
        for _ in range(2)
    ])
    assert select_candidate(table)["name"] == "b"
    assert select_candidate(table, focused=True)["selection_constraint_feasible"]
    table["prediabetes_precision"] = .01
    assert not select_candidate(table, focused=True)["selection_constraint_feasible"]


@pytest.mark.parametrize("strategy", ["random", "smotenc"])
def test_resampling_preserves_original_rows_and_categorical_codes(strategy):
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, size=(140, len(FEATURES))).astype(np.float32)
    x[:, FEATURES.index("BMI")] = rng.integers(20, 40, 140)
    x[:, FEATURES.index("MentHlth")] = rng.integers(0, 30, 140)
    y = np.array([0]*100+[1]*8+[2]*32)
    original = x.copy()
    xr, yr = resample_training(x, y, strategy, seed=42)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(xr[:len(x)], original)
    assert (yr == 1).sum() == 15
    assert (yr == 0).sum() == 100 and (yr == 2).sum() == 32
    numeric = {"BMI", "MentHlth", "PhysHlth"}
    for i, name in enumerate(FEATURES):
        if name not in numeric:
            assert np.isin(xr[:, i], np.unique(original[:, i])).all()


def test_neural_preprocessing_uses_fit_rows_only_and_model_roundtrips(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(150, len(FEATURES))).astype(np.float32)
    y = np.array([0, 1, 2]*50)
    roles = {"fit": np.arange(90), "stopping": np.arange(90, 120),
             "calibration": np.arange(120, 135), "decision": np.arange(135, 150)}
    x[90:] += 100
    config = {"kind": "mlp", "hidden": [8], "lr": .001, "dropout": 0.,
              "weight_decay": 0., "epochs": 1, "power": 0., "sampling": "none", "gamma": 0.}
    bundle = fit_model(x, y, roles, config)
    np.testing.assert_allclose(bundle["scaler"].mean_, x[:90].mean(0), atol=1e-6)
    cal = VectorCalibrator().fit(predict_raw(bundle, x[roles["calibration"]]), y[roles["calibration"]])
    save_bundle(bundle, cal, tmp_path)
    loaded, loaded_cal = load_bundle(tmp_path)
    np.testing.assert_allclose(loaded_cal.transform(predict_raw(loaded, x)), cal.transform(predict_raw(bundle, x)), atol=1e-6)


def test_completed_v2_keeps_historical_artifacts_and_generates_valid_predictions():
    import hashlib
    import json
    from diabetes_ml.common import ROOT
    from diabetes_ml.v2_protocol import OUT
    from diabetes_ml.v2_predict import predict
    if not (OUT / "summary.json").exists():
        pytest.skip("Run the v2 experiment first")
    history = json.loads((OUT / "historical_hashes.json").read_text())
    for name, checksum in history.items():
        path = ROOT / name
        if not path.exists():
            continue  # Intermediate arrays may be excluded from Git.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    manifest = json.loads((OUT / "role_audit.json").read_text())
    assert manifest["historical_test_rows_used"] == manifest["grouped_outer_profile_overlap"] == 0
    from diabetes_ml.common import f1_from_cm
    summary = pd.read_csv(OUT / "oof_summary.csv")
    for row in summary.itertuples():
        confusion = pd.read_csv(OUT / f"{row.method}_confusion.csv").to_numpy()
        assert confusion.sum() == 228312
        assert float(f1_from_cm(confusion)) == pytest.approx(row.macro_f1)
    assert len(pd.read_csv(OUT / "outer_results.csv")) == 18
    assert len(pd.read_csv(OUT / "seed_results.csv")) == 9
    if (OUT / "cache/role_row_ids.npz").exists():
        ids = np.load(OUT / "cache/role_row_ids.npz")
        splits = np.load(ROOT / "data/processed/splits.npz")
        for key in ids.files:
            assert not np.isin(ids[key], splits["test"]).any()
        groups = {key.rsplit("_",1)[0] for key in ids.files}
        for group in groups:
            arrays = [ids[f"{group}_{role}"] for role in ["fit", "stopping", "calibration", "decision", "evaluation"]]
            joined = np.concatenate(arrays)
            assert len(joined) == len(np.unique(joined))
    frame = pd.read_csv(ROOT / "data/processed/example_inputs.csv")
    for policy in ["macro", "recall"]:
        result = predict(frame, policy)
        assert result.shape == (len(frame), 5)
        np.testing.assert_allclose(result.filter(like="score_class_").sum(axis=1), 1, atol=1e-6)
        assert result.predicted_class.isin([0, 1, 2]).all()
