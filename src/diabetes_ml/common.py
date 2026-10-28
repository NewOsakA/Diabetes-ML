import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["No Diabetes", "Prediabetes", "Diabetes"]
TARGET = "Diabetes_012"
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def fast_macro_f1(y, pred):
    cm = np.bincount(3 * np.asarray(y, dtype=int) + pred, minlength=9).reshape(3, 3)
    return f1_from_cm(cm)


def f1_from_cm(cm):
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    return np.divide(2 * diag, den, out=np.zeros_like(diag, dtype=float), where=den != 0).mean(axis=-1)


def decision_predict(proba, offsets):
    """Cost-sensitive argmax. Offsets change decisions, not probability estimates."""
    return (np.log(np.clip(proba, 1e-12, 1)) + np.asarray(offsets)).argmax(axis=1)


def tune_offsets(y, proba):
    """Small predeclared grid on tuning data only; no use of selection/test labels."""
    best = (fast_macro_f1(y, proba.argmax(axis=1)), np.zeros(3))
    logp = np.log(np.clip(proba, 1e-12, 1))
    for b1 in np.arange(-1.0, 4.01, 0.25):
        for b2 in np.arange(-0.5, 2.01, 0.25):
            offsets = np.array([0.0, b1, b2])
            score = fast_macro_f1(y, (logp + offsets).argmax(axis=1))
            if score > best[0] + 1e-12:
                best = (float(score), offsets)
    return best[1].tolist(), float(best[0])


def evaluate(y, proba, offsets=(0, 0, 0)):
    pred = decision_predict(proba, offsets)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0
    )
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    specificity = np.divide(tn, tn + fp, out=np.zeros(3), where=(tn + fp) != 0)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "prediabetes_recall": float(recall[1]),
        "prediabetes_precision": float(precision[1]),
        "diabetes_recall": float(recall[2]),
    }
    rows = []
    for i, name in enumerate(CLASSES):
        rows.append({"class": name, "precision": float(precision[i]),
                     "recall": float(recall[i]), "specificity": float(specificity[i]),
                     "f1": float(f1[i]), "support": int(support[i]),
                     "average_precision": float(average_precision_score(np.asarray(y) == i, proba[:, i]))})
    return metrics, pd.DataFrame(rows), cm, pred


def paired_bootstrap(y, baseline, improved, n=2000, seed=42):
    """Stratified paired row bootstrap from exact joint-prediction contingency counts."""
    rng = np.random.default_rng(seed)
    base_cm = np.zeros((n, 3, 3), dtype=int)
    new_cm = np.zeros((n, 3, 3), dtype=int)
    for c in range(3):
        mask = np.asarray(y) == c
        counts = np.bincount(3 * baseline[mask] + improved[mask], minlength=9)
        joint = rng.multinomial(int(mask.sum()), counts / counts.sum(), size=n).reshape(n, 3, 3)
        base_cm[:, c, :] = joint.sum(axis=2)
        new_cm[:, c, :] = joint.sum(axis=1)
    delta = f1_from_cm(new_cm) - f1_from_cm(base_cm)
    return {"resamples": n, "seed": seed, "method": "paired stratified row bootstrap",
            "macro_f1_delta_95_ci": np.quantile(delta, [0.025, 0.975]).tolist(),
            "improved_macro_f1_95_ci": np.quantile(f1_from_cm(new_cm), [0.025, 0.975]).tolist(),
            "limitation": "Conditional on this split and fitted models; does not capture training-seed, model-selection, survey-design or profile-cluster uncertainty."}
