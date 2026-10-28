"""Batch inference with the frozen validation-selected model and decision rule."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch

from .common import ROOT, FEATURES, CLASSES, decision_predict
from .neural import MLP, predict_neural


class Predictor:
    def __init__(self, root=ROOT):
        self.root = root
        self.manifest = json.loads((root / "models/selection_manifest.json").read_text())
        self.registry = json.loads((root / "models/registry.json").read_text())
        self.scaler = joblib.load(root / "models/scaler.joblib")
        self.models = {}

    def raw_scores(self, frame, name=None):
        if not set(FEATURES).issubset(frame.columns):
            raise ValueError(f"Missing features: {sorted(set(FEATURES) - set(frame.columns))}")
        x = frame[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Inputs must be finite, non-missing numeric survey codes")
        if not np.equal(x, np.floor(x)).all():
            raise ValueError("Inputs must use the integer BRFSS survey codes")
        audit = json.loads((self.root / "results/data_audit.json").read_text())
        for i, feature in enumerate(FEATURES):
            low, high = audit["feature_ranges"][feature]
            if ((x[:, i] < low) | (x[:, i] > high)).any():
                raise ValueError(f"{feature} must be between {low:g} and {high:g}; see the data dictionary")
        name = name or self.manifest["champion"]["model"]
        spec = self.registry[name]
        if spec["kind"] == "ensemble":
            return sum(w * self.raw_scores(frame, member) for member, w in zip(spec["members"], spec["weights"]) if w)
        if name not in self.models:
            if spec["kind"] == "torch":
                saved = torch.load(self.root / "models" / spec["file"], map_location="cpu", weights_only=True)
                model = MLP(len(FEATURES), saved["config"]["hidden"], saved["config"].get("dropout", 0))
                model.load_state_dict(saved["state_dict"])
            else:
                model = joblib.load(self.root / "models" / spec["file"])
            self.models[name] = model
        model = self.models[name]
        if spec["kind"] == "torch":
            return predict_neural(model, self.scaler.transform(x).astype(np.float32))
        return model.predict_proba(self.scaler.transform(x) if spec["kind"] == "linear" else x)

    def predict(self, frame):
        p = self.raw_scores(frame)
        pred = decision_predict(p, self.manifest["champion"]["offsets"])
        output = pd.DataFrame({"predicted_class": pred, "predicted_label": [CLASSES[i] for i in pred]}, index=frame.index)
        for i in range(3):
            output[f"raw_score_class_{i}"] = p[:, i]
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing all 21 named BRFSS features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = Predictor().predict(pd.read_csv(args.input))
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
