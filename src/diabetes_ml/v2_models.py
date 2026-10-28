"""Fold-local fitting, resampling, independent calibration and inference."""
import copy

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from .common import FEATURES, fast_macro_f1
from .neural import MLP, classification_loss, predict_neural


class VectorCalibrator:
    """Positive per-class log-score scaling + bias, fitted on independent rows."""
    def fit(self, proba, y):
        logp = np.log(np.clip(proba, 1e-12, 1)).astype(float)
        def objective(theta):
            scaled = logp * np.exp(theta[:3]) + np.r_[0., theta[3:]]
            return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]).mean() + 1e-4 * (theta**2).sum())
        result = minimize(objective, np.zeros(5), method="L-BFGS-B", bounds=[(-3, 3)]*3+[(-8, 8)]*2)
        if not result.success:
            raise RuntimeError(f"Calibration did not converge: {result.message}")
        self.theta_ = result.x
        self.n_calibration_ = len(y)
        return self

    def transform(self, proba):
        return softmax(np.log(np.clip(proba, 1e-12, 1)) * np.exp(self.theta_[:3]) + np.r_[0., self.theta_[3:]], axis=1)


def resample_training(x, y, strategy, seed):
    """Training data only. Keep binary/ordinal answers as valid categories."""
    from imblearn.over_sampling import RandomOverSampler, SMOTENC
    counts = np.bincount(y, minlength=3)
    target = max(int(counts[1]), int(.15 * counts[0]))
    if target == counts[1]:
        return x, y
    sampling_strategy = {1: target}
    if strategy == "random":
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
    elif strategy == "smotenc":
        # BMI and day counts are numeric; all binary and ordinal survey codes categorical.
        numeric = {"BMI", "MentHlth", "PhysHlth"}
        categorical = [i for i, name in enumerate(FEATURES) if name not in numeric]
        sampler = SMOTENC(categorical_features=categorical, sampling_strategy=sampling_strategy,
                          random_state=seed, k_neighbors=min(5, counts[1]-1))
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")
    xr, yr = sampler.fit_resample(x, y)
    if strategy == "smotenc":
        for i in categorical:
            assert np.isin(xr[:, i], np.unique(x[:, i])).all()
    return np.asarray(xr, dtype=np.float32), np.asarray(yr, dtype=np.int64)


def fit_model(x, y, roles, config, seed=42):
    fit, stop = roles["fit"], roles["stopping"]
    counts = np.bincount(y[fit], minlength=3)
    weights = (len(fit) / (3 * counts)) ** config["power"]
    if config["kind"] == "catboost":
        model = CatBoostClassifier(depth=config["depth"], iterations=config["iterations"],
                    learning_rate=config["lr"], l2_leaf_reg=config["l2"], class_weights=weights.tolist(),
                    loss_function="MultiClass", eval_metric="TotalF1:average=Macro;use_weights=false",
                    early_stopping_rounds=70, random_seed=seed, thread_count=4,
                    allow_writing_files=False, verbose=False)
        model.fit(x[fit], y[fit], eval_set=(x[stop], y[stop]), use_best_model=True)
        return {"kind": "catboost", "model": model, "config": config, "seed": seed,
                "best_iteration": int(model.best_iteration_), "fit_rows": len(fit)}

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    scaler = StandardScaler().fit(x[fit])
    xr, yr = x[fit], y[fit]
    if config["sampling"] in ["random", "smotenc"]:
        xr, yr = resample_training(xr, yr, config["sampling"], seed)
    xt = torch.tensor(scaler.transform(xr), dtype=torch.float32)
    yt = torch.tensor(yr, dtype=torch.long)
    xs = scaler.transform(x[stop]).astype(np.float32)
    weights = weights / np.average(weights, weights=counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    model = MLP(len(FEATURES), config["hidden"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    if config["sampling"] == "balanced":
        # Replacement sampling makes classes equally frequent in expectation; no loss reweighting.
        sampling_weights = torch.tensor(1 / counts[yr], dtype=torch.double)
    best, best_epoch, state, history = -1., 0, None, []
    for epoch in range(1, config["epochs"]+1):
        model.train()
        order = (torch.multinomial(sampling_weights, len(yr), replacement=True, generator=generator)
                 if config["sampling"] == "balanced" else torch.randperm(len(yr), generator=generator))
        total = 0.
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(xt[ids]), yt[ids], weight_tensor, config["gamma"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        p = predict_neural(model, xs)
        score = float(fast_macro_f1(y[stop], p.argmax(1)))
        history.append({"epoch": epoch, "train_objective": total/len(yr),
                        "stopping_macro_f1": score, "stopping_log_loss": float(log_loss(y[stop], p))})
        if score > best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return {"kind": "mlp", "model": model, "scaler": scaler, "config": config, "seed": seed,
            "best_epoch": best_epoch, "history": history, "fit_rows": len(fit), "resampled_rows": len(yr)}


def predict_raw(bundle, x):
    if bundle["kind"] == "mlp":
        return predict_neural(bundle["model"], bundle["scaler"].transform(x).astype(np.float32))
    return bundle["model"].predict_proba(x)


def fit_calibrator(bundle, x, y, roles):
    ids = roles["calibration"]
    return VectorCalibrator().fit(predict_raw(bundle, x[ids]), y[ids])


def save_bundle(bundle, calibrator, path):
    path.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in bundle.items() if k not in ["model", "scaler", "history"]}
    joblib.dump({"metadata": metadata, "calibrator": calibrator}, path / "metadata.joblib")
    if bundle["kind"] == "mlp":
        torch.save(bundle["model"].state_dict(), path / "weights.pt")
        joblib.dump(bundle["scaler"], path / "scaler.joblib")
        pd.DataFrame(bundle["history"]).to_csv(path / "history.csv", index=False)
    else:
        bundle["model"].save_model(str(path / "model.cbm"))


def load_bundle(path):
    saved = joblib.load(path / "metadata.joblib")
    bundle = saved["metadata"]
    config = bundle["config"]
    if bundle["kind"] == "mlp":
        model = MLP(len(FEATURES), config["hidden"], config["dropout"])
        model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
        bundle["scaler"] = joblib.load(path / "scaler.joblib")
    else:
        model = CatBoostClassifier()
        model.load_model(str(path / "model.cbm"))
    bundle["model"] = model
    return bundle, saved["calibrator"]
