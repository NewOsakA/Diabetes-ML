"""Run with python -m diabetes_ml.train. Test labels are opened only after freeze."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import (ROOT, FEATURES, TARGET, write_json, evaluate, tune_offsets,
                     decision_predict, paired_bootstrap)
from .data import prepare
from .neural import train_neural, predict_neural
from . import plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun an already-completed experiment")
    args = parser.parse_args()
    if (ROOT / "results/test_results.csv").exists() and not args.overwrite:
        raise SystemExit("Completed test results exist. Use a new output directory for further tuning, or --overwrite to explicitly rerun the saved experiment.")
    start = time.time()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    for folder in ["models", "results/figures", "results/histories", "results/probabilities", "data/processed"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    df, splits, feature_hash = prepare(seed)
    plots.exploration(df)
    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)
    tr, tu, se = [splits[k] for k in ["train", "tuning", "selection"]]
    scaler = StandardScaler().fit(x[tr])
    joblib.dump(scaler, ROOT / "models/scaler.joblib")
    xs = {k: scaler.transform(x[idx]).astype(np.float32) for k, idx in splits.items() if k != "test"}
    counts = np.bincount(y[tr], minlength=3)
    weights = len(tr) / (3 * counts)
    registry, prob, rows = {}, {}, []
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "torch", "catboost", "scipy"]},
                   "seed": seed, "threads": 4, "device": "cpu", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(ROOT / "results/environment.json", environment)

    def register(name, model, kind, config, elapsed):
        metadata = {"kind": kind, "config": config, "fit_seconds": elapsed}
        if kind == "torch":
            filename = f"{name}.pt"
            torch.save({"state_dict": model.state_dict(), "config": config, "features": FEATURES}, ROOT / "models" / filename)
            ps = {k: predict_neural(model, xs[k]) for k in ["tuning", "selection"]}
        else:
            filename = f"{name}.joblib"
            joblib.dump(model, ROOT / "models" / filename)
            ps = {k: model.predict_proba(xs[k] if kind == "linear" else x[splits[k]]) for k in ["tuning", "selection"]}
        metadata["file"] = filename
        registry[name] = metadata
        prob[name] = ps
        for split, p in ps.items():
            np.save(ROOT / f"results/probabilities/{name}_{split}.npy", p)
        for decision in ["argmax", "tuned"]:
            offsets = [0., 0., 0.] if decision == "argmax" else tune_offsets(y[tu], ps["tuning"])[0]
            for split, p in ps.items():
                metrics, per_class, cm, _ = evaluate(y[splits[split]], p, offsets)
                rows.append({"model": name, "decision": decision, "split": split, **metrics,
                             "fit_seconds": elapsed, "offsets": json.dumps(offsets)})
                per_class.to_csv(ROOT / f"results/{name}_{decision}_{split}_classes.csv", index=False)
            print(f"{name} {decision}: selection macro F1={rows[-1]['macro_f1']:.4f}, accuracy={rows[-1]['accuracy']:.4f}, prediabetes recall={rows[-1]['prediabetes_recall']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results/experiment_results.csv", index=False)
        write_json(ROOT / "models/registry.json", registry)

    dummy = DummyClassifier(strategy="most_frequent").fit(x[tr], y[tr])
    register("dummy", dummy, "tree", {"strategy": "most_frequent"}, 0)
    for c, weighted in [(1., False), (.1, True), (1., True), (10., True)]:
        name = "logistic_baseline" if not weighted else f"logistic_weighted_C{c:g}"
        config = {"C": c, "class_weight": "balanced" if weighted else None, "max_iter": 1500, "solver": "lbfgs"}
        t = time.time()
        model = LogisticRegression(**config, random_state=seed).fit(xs["train"], y[tr])
        register(name, model, "linear", config, time.time() - t)
        if not weighted:
            plots.coefficients(model, scaler, FEATURES)

    neural_configs = [
        {"name": "mlp_baseline", "hidden": [64, 32], "lr": .001, "epochs": 30, "batch_size": 256},
        {"name": "mlp_sqrt_weighted", "hidden": [64, 32], "lr": .001, "epochs": 30, "batch_size": 256,
         "class_weight_power": .5, "dropout": .1, "weight_decay": .0001},
        {"name": "mlp_balanced", "hidden": [128, 64], "lr": .0005, "epochs": 30, "batch_size": 256,
         "class_weight_power": 1., "dropout": .2, "weight_decay": .001},
        {"name": "mlp_focal", "hidden": [128, 64], "lr": .001, "epochs": 30, "batch_size": 256,
         "class_weight_power": .5, "dropout": .15, "weight_decay": .0001, "focal_gamma": 2.},
    ]
    for config in neural_configs:
        t = time.time()
        model, history, fit_info = train_neural(xs["train"], y[tr], xs["tuning"], y[tu], config, seed)
        history.to_csv(ROOT / f"results/histories/{config['name']}.csv", index=False)
        plots.history_plot(history, config["name"])
        register(config["name"], model, "torch", {**config, **fit_info}, time.time() - t)

    for power in [0., .5, 1.]:
        name = f"hist_gradient_weight{power:g}"
        config = {"max_iter": 300, "learning_rate": .06, "max_leaf_nodes": 15,
                  "l2_regularization": 5., "early_stopping": True, "validation_fraction": .1,
                  "random_state": seed, "class_weight_power": power}
        t = time.time()
        kwargs = {k: v for k, v in config.items() if k != "class_weight_power"}
        model = HistGradientBoostingClassifier(**kwargs).fit(x[tr], y[tr], sample_weight=weights[y[tr]] ** power)
        register(name, model, "tree", {**config, "actual_iterations": int(model.n_iter_)}, time.time() - t)

    for depth, power, lr in [(5, 0., .06), (6, .5, .06), (7, .5, .04), (6, 1., .06)]:
        name = f"catboost_depth{depth}_weight{power:g}"
        config = {"depth": depth, "class_weight_power": power, "learning_rate": lr,
                  "iterations": 900, "l2_leaf_reg": 5., "random_seed": seed,
                  "loss_function": "MultiClass", "eval_metric": "TotalF1:average=Macro;use_weights=false",
                  "class_weights": (weights ** power).tolist(), "thread_count": 4,
                  "allow_writing_files": False, "verbose": 150, "early_stopping_rounds": 100}
        t = time.time()
        model = CatBoostClassifier(**{k: v for k, v in config.items() if k != "class_weight_power"})
        model.fit(x[tr], y[tr], eval_set=(x[tu], y[tu]), use_best_model=True)
        register(name, model, "tree", {**config, "best_iteration": int(model.best_iteration_)}, time.time() - t)
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(ROOT / f"results/{name}_importance.csv", index=False)

    # A predeclared soft-voting family: choose members using tuning data only.
    frame = pd.DataFrame(rows)
    tune_rank = frame[(frame.split == "tuning") & (frame.decision == "tuned")].sort_values("macro_f1", ascending=False)
    tree_names = [n for n in tune_rank.model if n.startswith(("catboost", "hist_gradient"))][:2]
    mlp_name = next(n for n in tune_rank.model if n.startswith("mlp"))
    members = tree_names + [mlp_name]
    for blend_weights in [[.5, .5, 0.], [.4, .4, .2]]:
        name = "ensemble_trees" if blend_weights[-1] == 0 else "ensemble_trees_mlp"
        ps = {k: sum(w * prob[n][k] for n, w in zip(members, blend_weights)) for k in ["tuning", "selection"]}
        registry[name] = {"kind": "ensemble", "members": members, "weights": blend_weights, "fit_seconds": 0}
        prob[name] = ps
        for decision in ["argmax", "tuned"]:
            offsets = [0., 0., 0.] if decision == "argmax" else tune_offsets(y[tu], ps["tuning"])[0]
            for split, p in ps.items():
                metrics, _, _, _ = evaluate(y[splits[split]], p, offsets)
                rows.append({"model": name, "decision": decision, "split": split, **metrics,
                             "fit_seconds": 0., "offsets": json.dumps(offsets)})
        print(f"{name}: tuned selection macro F1={rows[-1]['macro_f1']:.4f}", flush=True)

    experiments = pd.DataFrame(rows)
    experiments.to_csv(ROOT / "results/experiment_results.csv", index=False)
    write_json(ROOT / "models/registry.json", registry)
    candidates = experiments[experiments.split == "selection"].sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False, kind="stable")
    champion = candidates.iloc[0]
    # Freeze selection, member weights and offsets BEFORE test inference/evaluation.
    finalists = []
    def finalist(label, row):
        finalists.append({"label": label, "model": row.model, "decision": row.decision,
                          "offsets": json.loads(row.offsets), "selection_macro_f1": float(row.macro_f1)})
    for base in ["dummy", "logistic_baseline", "mlp_baseline"]:
        finalist(base, candidates[(candidates.model == base) & (candidates.decision == "argmax")].iloc[0])
    finalist("best_weighted_logistic", candidates[candidates.model.str.startswith("logistic_weighted")].iloc[0])
    finalist("best_improved_mlp", candidates[candidates.model.isin([c["name"] for c in neural_configs[1:]])].iloc[0])
    finalist("best_boosted_tree", candidates[candidates.model.str.startswith(("catboost", "hist_gradient"))].iloc[0])
    finalist("champion", champion)
    # Same fitted champion with ordinary argmax isolates the decision-rule contribution.
    raw = candidates[(candidates.model == champion.model) & (candidates.decision == "argmax")].iloc[0]
    finalist("champion_argmax_ablation", raw)
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_metric": "macro_f1",
                "tie_breaker": "balanced_accuracy", "champion": finalists[-2], "finalists": finalists,
                "features": FEATURES, "seed": seed, "test_used_for_selection": False,
                "source_sha256": hashlib.sha256((ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv").read_bytes()).hexdigest()}
    write_json(ROOT / "models/selection_manifest.json", manifest)
    print("SELECTION FROZEN:", manifest["champion"], flush=True)

    test = splits["test"]
    xt = scaler.transform(x[test]).astype(np.float32)
    test_prob = {}
    def get_test_proba(name):
        if name in test_prob:
            return test_prob[name]
        spec = registry[name]
        if spec["kind"] == "ensemble":
            p = sum(w * get_test_proba(n) for n, w in zip(spec["members"], spec["weights"]) if w)
        elif spec["kind"] == "torch":
            from .neural import MLP
            saved = torch.load(ROOT / "models" / spec["file"], map_location="cpu", weights_only=True)
            model = MLP(len(FEATURES), saved["config"]["hidden"], saved["config"].get("dropout", 0))
            model.load_state_dict(saved["state_dict"])
            p = predict_neural(model, xt)
        else:
            model = joblib.load(ROOT / "models" / spec["file"])
            p = model.predict_proba(xt if spec["kind"] == "linear" else x[test])
        test_prob[name] = p
        return p

    test_rows, predictions = [], {}
    for item in finalists:
        p = get_test_proba(item["model"])
        metrics, per_class, cm, pred = evaluate(y[test], p, item["offsets"])
        label = item["label"]
        predictions[label] = pred
        test_rows.append({"model": label, "underlying_model": item["model"], "decision": item["decision"], **metrics})
        per_class.to_csv(ROOT / f"results/{label}_test_classes.csv", index=False)
        pd.DataFrame(cm).to_csv(ROOT / f"results/{label}_test_confusion.csv", index=False)
        plots.confusion(cm, label, "test")
        # Summarize the combined validation split (descriptive only).
        pv = np.concatenate([prob[item["model"]]["tuning"], prob[item["model"]]["selection"]])
        yv = np.concatenate([y[tu], y[se]])
        vm, vc, vcm, _ = evaluate(yv, pv, item["offsets"])
        vc.to_csv(ROOT / f"results/{label}_validation_classes.csv", index=False)
        plots.confusion(vcm, label, "validation")
        write_json(ROOT / f"results/{label}_validation_metrics.json", vm)
        if label == "champion":
            plots.precision_recall(y[test], p, item["model"])
            pd.DataFrame({"row_id": test, "true_class": y[test], "predicted_class": pred,
                          **{f"raw_score_class_{i}": p[:, i] for i in range(3)}}).to_csv(ROOT / "results/champion_test_predictions.csv", index=False)
        print(f"TEST {label}: {metrics}", flush=True)
    table = pd.DataFrame(test_rows)
    table.to_csv(ROOT / "results/test_results.csv", index=False)
    plots.comparison(table)
    for baseline in ["mlp_baseline", "logistic_baseline"]:
        ci = paired_bootstrap(y[test], predictions[baseline], predictions["champion"], seed=seed)
        write_json(ROOT / f"results/bootstrap_vs_{baseline}.json", ci)
    seen = np.isin(feature_hash[test], np.unique(feature_hash[tr]))
    sensitivity = []
    for label in ["mlp_baseline", "champion"]:
        item = next(i for i in finalists if i["label"] == label)
        for subset, mask in [("profile_seen_in_training", seen), ("profile_unseen_in_training", ~seen)]:
            metrics, classes, _, _ = evaluate(y[test][mask], get_test_proba(item["model"])[mask], item["offsets"])
            sensitivity.append({"model": label, "subset": subset, "rows": int(mask.sum()), **metrics})
            classes.to_csv(ROOT / f"results/{label}_{subset}_classes.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT / "results/profile_sensitivity.csv", index=False)
    write_json(ROOT / "results/run_summary.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start, "candidate_model_count": len(registry),
                "decision_variants": len(candidates), "champion": manifest["champion"]})
    print(f"Completed in {(time.time()-start)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
