"""Fixed development-only protocol; historic test IDs never enter experiments."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, FEATURES, TARGET, fast_macro_f1

OUT = ROOT / "experiments/v2"
SEED = 42
PRECISION_FLOOR = 0.05  # Research operating point, not a clinical requirement.
MODES = ["raw_argmax", "raw_macro", "calibrated_argmax", "calibrated_macro",
         "raw_recall", "calibrated_recall"]


def candidates():
    base = {"kind": "mlp", "hidden": [64, 32], "lr": .001, "dropout": .1,
            "weight_decay": .0001, "epochs": 20, "power": .5, "sampling": "none", "gamma": 0.}
    configs = []
    def add(name, **kwargs):
        configs.append({**base, "name": name, **kwargs})
    add("fixed_plain_mlp", power=0., dropout=0., weight_decay=0., epochs=30)
    add("fixed_weighted_mlp", epochs=30)
    add("mlp_small_p025", hidden=[32, 16], power=.25)
    add("mlp_regularized_p075", power=.75, lr=.0005, dropout=.2, weight_decay=.001)
    add("mlp_wide_p1", hidden=[128, 64], power=1., lr=.0005, dropout=.2)
    add("mlp_focal_g1", power=.25, gamma=1.)
    add("mlp_focal_g2", hidden=[128, 64], gamma=2., dropout=.15)
    add("mlp_balanced_sampling", power=0., sampling="balanced")
    add("mlp_random_oversampling", power=0., sampling="random")
    add("mlp_smotenc", power=0., sampling="smotenc")
    for i, (depth, power, lr, l2) in enumerate([
        (4, 0., .06, 5), (5, .25, .1, 3), (6, .5, .06, 5),
        (7, .5, .04, 10), (5, .75, .06, 10), (6, 1., .04, 5),
        (4, .5, .1, 3), (6, .25, .03, 10),
    ]):
        configs.append({"name": f"catboost_{i}", "kind": "catboost", "depth": depth,
                        "power": power, "lr": lr, "l2": l2, "iterations": 700})
    return configs


def load_development():
    splits = np.load(ROOT / "data/processed/splits.npz")
    ids = np.sort(np.concatenate([splits[k] for k in ["train", "tuning", "selection"]]))
    historic = splits["test"]
    assert len(ids) == len(np.unique(ids)) == 228312
    assert not np.intersect1d(ids, historic).size
    path = ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"
    source = json.loads((ROOT / "data/raw/source.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    # Discard historic test rows before any analysis, fitting or scoring.
    dev = pd.read_csv(path).iloc[ids].reset_index(drop=True)
    return dev, ids, historic


def partition_roles(ids, y, seed=SEED):
    """70% fit / 10% stopping / 10% calibration / 10% decisions, disjoint."""
    ids = np.asarray(ids)
    fit, auxiliary = train_test_split(ids, test_size=.3, stratify=y[ids], random_state=seed)
    stop, remaining = train_test_split(auxiliary, test_size=2/3, stratify=y[auxiliary], random_state=seed+1)
    calibration, decision = train_test_split(remaining, test_size=.5, stratify=y[remaining], random_state=seed+2)
    return {"fit": fit, "stopping": stop, "calibration": calibration, "decision": decision}


def validate_roles(roles, evaluation, development_ids, historic_ids):
    """All inputs use original source row IDs, so accidental remapping is caught."""
    arrays = list(roles.values()) + [np.asarray(evaluation)]
    all_ids = np.concatenate(arrays)
    if len(all_ids) != len(np.unique(all_ids)):
        raise ValueError("Training roles and evaluation rows must be disjoint")
    if not np.isin(all_ids, development_ids).all() or np.isin(all_ids, historic_ids).any():
        raise ValueError("Historical test IDs must never enter the v2 experiment")


def tune_decisions(y, proba, floor=PRECISION_FLOOR, min_predictions=30):
    """Choose macro-F1 and recall-focused rules using decision rows only."""
    logp = np.log(np.clip(proba, 1e-12, 1))
    macro_best = (float(fast_macro_f1(y, proba.argmax(1))), [0., 0., 0.])
    recall_best = None
    for b1 in np.arange(-1., 5.01, .25):
        for b2 in np.arange(-.5, 2.01, .25):
            offsets = [0., float(b1), float(b2)]
            pred = (logp + offsets).argmax(1)
            cm = np.bincount(3*y+pred, minlength=9).reshape(3, 3)
            macro = float(fast_macro_f1(y, pred))
            if macro > macro_best[0]:
                macro_best = (macro, offsets)
            n_positive = int(cm[:, 1].sum())
            precision = cm[1, 1] / max(1, n_positive)
            recall = cm[1, 1] / max(1, cm[1].sum())
            if n_positive >= min_predictions and precision >= floor and macro >= .4:
                key = (float(recall), macro, float(precision))
                if recall_best is None or key > recall_best[0]:
                    recall_best = (key, offsets)
    return {"macro": macro_best[1], "recall": recall_best[1] if recall_best else macro_best[1],
            "recall_constraint_feasible": recall_best is not None}
