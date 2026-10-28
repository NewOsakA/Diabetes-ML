"""Inference with a development-trained v2 model; original predictor is unchanged."""
import argparse
import json

import numpy as np
import pandas as pd

from .common import ROOT, FEATURES, CLASSES, decision_predict
from .v2_protocol import OUT
from .v2_models import load_bundle, predict_raw


def predict(frame, policy="macro", model_directory=None):
    if policy not in ["macro", "recall"]:
        raise ValueError("Policy must be macro or recall")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    x = frame[FEATURES].to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise ValueError("Inputs must be finite integer BRFSS survey codes")
    ranges = json.loads((ROOT / "results/data_audit.json").read_text())["feature_ranges"]
    for i, feature in enumerate(FEATURES):
        low, high = ranges[feature]
        if ((x[:, i] < low) | (x[:, i] > high)).any():
            raise ValueError(f"{feature} must be between {low:g} and {high:g}")
    directory = model_directory or OUT / f"models/{policy}"
    bundle, calibrator = load_bundle(directory)
    rule = json.loads((directory / "decision.json").read_text())
    p = predict_raw(bundle, x)
    if rule["mode"].startswith("calibrated"):
        p = calibrator.transform(p)
    pred = decision_predict(p, rule["offsets"])
    return pd.DataFrame({"predicted_class": pred, "predicted_label": [CLASSES[i] for i in pred],
                         **{f"score_class_{i}": p[:, i] for i in range(3)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=["macro", "recall"], default="macro")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.input), args.policy)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} v2 predictions to {args.output}")


if __name__ == "__main__":
    main()
