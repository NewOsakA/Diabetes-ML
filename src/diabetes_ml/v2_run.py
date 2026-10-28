"""Development-only nested CV. Run: python -m diabetes_ml.v2_run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .common import ROOT, FEATURES, TARGET, write_json, evaluate, decision_predict
from .v2_models import fit_model, predict_raw, fit_calibrator, save_bundle
from .v2_protocol import (OUT, SEED, PRECISION_FLOOR, MODES, candidates, load_development,
                          partition_roles, validate_roles, tune_decisions)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p, offsets):
    result = evaluate(y, p, offsets)[0]
    result["prediabetes_ap"] = float(average_precision_score(y == 1, p[:, 1]))
    result["log_loss"] = float(log_loss(y, p))
    result["brier"] = float(((p - np.eye(3)[y])**2).sum(axis=1).mean())
    return result


def select_candidate(table, family=None, focused=False):
    if family:
        table = table[table.kind == family]
    columns = ["name", "kind", "mode"]
    average = table.groupby(columns, as_index=False).mean(numeric_only=True)
    if focused:
        feasible = average[average["mode"].str.endswith("recall") &
                           (average.prediabetes_precision >= PRECISION_FLOOR) & (average.macro_f1 >= .4)]
        if len(feasible):
            result = feasible.sort_values(["prediabetes_recall", "macro_f1"], ascending=False, kind="stable").iloc[0].to_dict()
            result["selection_constraint_feasible"] = True
            return result
    result = average.sort_values(["macro_f1", "balanced_accuracy"], ascending=False, kind="stable").iloc[0].to_dict()
    result["selection_constraint_feasible"] = not focused
    return result


def score_bundle(bundle, x, y, roles, evaluation):
    cal = fit_calibrator(bundle, x, y, roles)
    decision_raw = predict_raw(bundle, x[roles["decision"]])
    evaluation_raw = predict_raw(bundle, x[evaluation])
    predictions, rows, rules = {}, [], {}
    for prefix in ["raw", "calibrated"]:
        pd = decision_raw if prefix == "raw" else cal.transform(decision_raw)
        pe = evaluation_raw if prefix == "raw" else cal.transform(evaluation_raw)
        tuned = tune_decisions(y[roles["decision"]], pd)
        for policy in ["argmax", "macro", "recall"]:
            mode = f"{prefix}_{policy}"
            offsets = [0., 0., 0.] if policy == "argmax" else tuned[policy]
            rules[mode] = {"offsets": offsets, "decision_constraint_feasible": tuned["recall_constraint_feasible"] if policy == "recall" else True}
            predictions[mode] = (pe, decision_predict(pe, offsets))
            rows.append({"mode": mode, **metrics(y[evaluation], pe, offsets), **rules[mode], "offsets": json.dumps(offsets)})
    return rows, predictions, rules, cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reuse completed inner folds with the identical protocol")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    if (OUT / "summary.json").exists():
        raise SystemExit("Completed v2 results exist. Preserve them and use a new experiment directory for changed protocols.")
    start = time.time()
    dev, source_ids, historical_ids = load_development()
    x, y = dev[FEATURES].to_numpy(np.float32), dev[TARGET].to_numpy(np.int64)
    configs = candidates()
    by_name = {c["name"]: c for c in configs}
    protocol = {"seed": SEED, "outer_folds": 3, "inner_folds": 2, "training_seeds": [42, 43, 44],
                "development_rows": len(dev), "excluded_historical_test_rows": len(historical_ids),
                "precision_floor": PRECISION_FLOOR, "minimum_decision_positive_predictions": 30,
                "decision_macro_f1_floor": .4, "role_ratios": {"fit": .7, "stopping": .1, "calibration": .1, "decision": .1},
                "candidates": configs, "modes": MODES, "source_sha256": digest(ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv"),
                "source_code_sha256": {f"{name}.py": digest(ROOT / f"src/diabetes_ml/{name}.py") for name in ["v2_run", "v2_protocol", "v2_models", "neural", "common"]},
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "torch", "scikit-learn", "catboost", "imbalanced-learn", "scipy"]},
                "interpretation": "Development-stage nested-CV estimates. Dataset and candidate ideas were previously explored; no new external test set is claimed."}
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit("Protocol already exists. --resume requires exactly the same code and configuration.")
    else:
        write_json(protocol_path, protocol)
        write_json(OUT / "started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat()})
    # Protect every original result and model against accidental overwrite.
    old_files = list((ROOT / "models").rglob("*")) + list((ROOT / "results").rglob("*")) + [ROOT / "data/processed/splits.npz"]
    historical_hashes = {str(p.relative_to(ROOT)): digest(p) for p in old_files if p.is_file()}
    write_json(OUT / "historical_hashes.json", historical_hashes)
    role_manifests = {}
    all_inner, outer_rows, calibration_rows, seed_rows = [], [], [], []
    oof = {}
    fold_choices = []
    outer_assignments = np.full(len(dev), -1, dtype=int)

    def role_split(training, evaluation, key, seed):
        roles = partition_roles(training, y, seed)
        validate_roles({k: source_ids[v] for k, v in roles.items()}, source_ids[evaluation], source_ids, historical_ids)
        for role, ids in roles.items():
            role_manifests[f"{key}_{role}"] = source_ids[ids]
        role_manifests[f"{key}_evaluation"] = source_ids[evaluation]
        return roles

    def search(training, key, split_seed):
        inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=split_seed)
        rows = []
        for inner_fold, (train_local, val_local) in enumerate(inner.split(x[training], y[training])):
            ti, vi = training[train_local], training[val_local]
            roles = role_split(ti, vi, f"{key}_inner{inner_fold}", split_seed+inner_fold)
            for config in configs:
                cache = OUT / f"cache/{key}_inner{inner_fold}_{config['name']}.json"
                if args.resume and cache.exists():
                    scored = json.loads(cache.read_text())
                else:
                    tick = time.time()
                    bundle = fit_model(x, y, roles, config, SEED)
                    scored, _, _, _ = score_bundle(bundle, x, y, roles, vi)
                    scored = [{"name": config["name"], "kind": config["kind"], "search": key, "inner_fold": inner_fold,
                               "fit_seconds": time.time()-tick, **r} for r in scored]
                    write_json(cache, scored)
                    print(f"{key} inner {inner_fold+1}/2 {config['name']}: {time.time()-tick:.1f}s", flush=True)
                rows.extend(scored)
        all_inner.extend(rows)
        pd.DataFrame(all_inner).to_csv(OUT / "inner_results.csv", index=False)
        return pd.DataFrame(rows)

    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (training, evaluation) in enumerate(outer.split(x, y)):
        print(f"START OUTER FOLD {fold+1}/3", flush=True)
        inner_scores = search(training, f"outer{fold}", SEED+100+fold)
        choice_mlp = select_candidate(inner_scores, "mlp")
        choice_cat = select_candidate(inner_scores, "catboost")
        choice_focus = select_candidate(inner_scores, focused=True)
        fixed_plain = {"name": "fixed_plain_mlp", "mode": "raw_argmax"}
        fixed_weighted = {"name": "fixed_weighted_mlp", "mode": "raw_macro"}
        choice_overall = select_candidate(inner_scores)
        choices = {"fixed_plain_mlp": fixed_plain, "fixed_weighted_mlp": fixed_weighted,
                   "selected_mlp": choice_mlp, "selected_catboost": choice_cat,
                   "prediabetes_focused": choice_focus, "selected_overall": choice_overall}
        write_json(OUT / f"fold{fold}_frozen_choices.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                   "choices": choices, "outer_evaluation_not_used_for_selection": True})
        roles = role_split(training, evaluation, f"outer{fold}_refit", SEED+500+fold)
        cache_fits = {}
        for method, choice in choices.items():
            name, mode = choice["name"], choice["mode"]
            if name not in cache_fits:
                bundle = fit_model(x, y, roles, by_name[name], SEED)
                rows, predictions, rules, cal = score_bundle(bundle, x, y, roles, evaluation)
                cache_fits[name] = bundle, rows, predictions, rules, cal
                for r in rows:
                    if r["mode"].endswith("argmax"):
                        calibration_rows.append({"fold": fold, "name": name, **r})
            bundle, rows, predictions, rules, cal = cache_fits[name]
            p, pred = predictions[mode]
            row = next(r for r in rows if r["mode"] == mode)
            outer_rows.append({"method": method, "fold": fold, "name": name, "seed": SEED, **row})
            if method not in oof:
                oof[method] = {"proba": np.zeros((len(y), 3)), "pred": np.zeros(len(y), dtype=int)}
            oof[method]["proba"][evaluation] = p
            oof[method]["pred"][evaluation] = pred
            print(f"OUTER {fold+1} {method}: macroF1={row['macro_f1']:.4f}, pre recall={row['prediabetes_recall']:.3f}, precision={row['prediabetes_precision']:.3f}", flush=True)
        # Seeds change stochastic training only; row-role assignments remain fixed.
        for seed in [42, 43, 44]:
            if seed == 42:
                _, rows, _, _, _ = cache_fits[choice_mlp["name"]]
            else:
                bundle = fit_model(x, y, roles, by_name[choice_mlp["name"]], seed)
                rows, _, _, _ = score_bundle(bundle, x, y, roles, evaluation)
            chosen_row = next(r for r in rows if r["mode"] == choice_mlp["mode"])
            seed_rows.append({"fold": fold, "seed": seed, "name": choice_mlp["name"], **chosen_row})
        fold_choices.append({"fold": fold, "choices": choices})
        outer_assignments[evaluation] = fold
        pd.DataFrame(outer_rows).to_csv(OUT / "outer_results.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(OUT / "seed_results.csv", index=False)
        pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_results.csv", index=False)
        np.savez_compressed(OUT / "cache/oof_partial.npz", row_id=source_ids, y=y, fold=outer_assignments,
                            **{f"{method}_{key}": value for method, values in oof.items() for key, value in values.items()})

    assert (outer_assignments >= 0).all()
    pd.DataFrame(outer_rows).groupby("method").agg({c: ["mean", "std"] for c in
        ["accuracy", "macro_f1", "balanced_accuracy", "prediabetes_recall", "prediabetes_precision", "prediabetes_ap", "log_loss", "brier"]}).to_csv(OUT / "outer_summary.csv")
    # Analysis of development-only, out-of-fold predictions and feature-profile conflicts.
    profiles = pd.util.hash_pandas_object(dev[FEATURES], index=False).to_numpy()
    profile_table = pd.DataFrame({"profile": profiles, "label": y})
    label_counts = profile_table.groupby("profile").label.nunique()
    conflicting = label_counts[label_counts > 1].index
    conflict_mask = np.isin(profiles, conflicting)
    write_json(OUT / "profile_audit.json", {"development_rows": len(dev), "unique_feature_profiles": int(len(label_counts)),
               "profiles_with_multiple_labels": int(len(conflicting)), "rows_in_conflicting_profiles": int(conflict_mask.sum()),
               "prediabetes_rows_in_conflicting_profiles": int(((y == 1) & conflict_mask).sum()),
               "interpretation": "Identical coarse survey responses can occur for different people; conflicting labels do not by themselves establish annotation errors."})
    subgroup_rows, aggregate_rows = [], []
    for method, values in oof.items():
        p, pred = values["proba"], values["pred"]
        # Preserve independently tuned fold-specific decisions, not a new pooled decision rule.
        _, per_class, cm, _ = evaluate(y, p)
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, f1_score
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        per_class["precision"], per_class["recall"], per_class["f1"] = precision, recall, f1
        fp, fn, tp = cm.sum(0)-cm.diagonal(), cm.sum(1)-cm.diagonal(), cm.diagonal()
        tn = cm.sum()-fp-fn-tp
        per_class["specificity"] = tn/(tn+fp)
        per_class.to_csv(OUT / f"{method}_classes.csv", index=False)
        pd.DataFrame(cm).to_csv(OUT / f"{method}_confusion.csv", index=False)
        aggregate_rows.append({"method": method, "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "prediabetes_recall": float(recall[1]), "prediabetes_precision": float(precision[1]),
            "diabetes_recall": float(recall[2]), "prediabetes_ap": float(average_precision_score(y == 1, p[:, 1]))})
        subgroup_masks = {"sex_0": dev.Sex.eq(0).to_numpy(), "sex_1": dev.Sex.eq(1).to_numpy(),
                          "age_1_to_5": dev.Age.le(5).to_numpy(), "age_6_to_9": dev.Age.between(6, 9).to_numpy(),
                          "age_10_to_13": dev.Age.ge(10).to_numpy(), "income_1_to_3": dev.Income.le(3).to_numpy(),
                          "income_4_to_6": dev.Income.between(4, 6).to_numpy(), "income_7_to_8": dev.Income.ge(7).to_numpy(),
                          "conflicting_profile": conflict_mask, "nonconflicting_profile": ~conflict_mask}
        for subgroup, mask in subgroup_masks.items():
            prec, rec, fone, sup = precision_recall_fscore_support(y[mask], pred[mask], labels=[0, 1, 2], zero_division=0)
            subgroup_rows.append({"method": method, "subgroup": subgroup, "rows": int(mask.sum()), "prediabetes_support": int(sup[1]),
                        "prediabetes_prevalence": float((y[mask] == 1).mean()), "prediabetes_precision": float(prec[1]),
                        "prediabetes_recall": float(rec[1]), "macro_f1": float(fone.mean()),
                        "prediabetes_ap": float(average_precision_score(y[mask] == 1, p[mask, 1]))})
    pd.DataFrame(aggregate_rows).to_csv(OUT / "oof_summary.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT / "subgroup_results.csv", index=False)

    # A separate sensitivity experiment: fixed weighted recipe with feature-profile-grouped folds.
    grouped_rows = []
    group_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (training, evaluation) in enumerate(group_cv.split(x, y, groups=profiles)):
        assert not np.intersect1d(profiles[training], profiles[evaluation]).size
        roles = role_split(training, evaluation, f"grouped{fold}", SEED+700+fold)
        bundle = fit_model(x, y, roles, by_name["fixed_weighted_mlp"], SEED)
        rows, _, _, _ = score_bundle(bundle, x, y, roles, evaluation)
        row = next(r for r in rows if r["mode"] == "raw_macro")
        grouped_rows.append({"fold": fold, "evaluation_rows": len(evaluation), "overlapping_profiles": 0, **row})
        print(f"GROUPED FOLD {fold+1}: macroF1={row['macro_f1']:.4f}", flush=True)
    pd.DataFrame(grouped_rows).to_csv(OUT / "grouped_sensitivity.csv", index=False)

    # Deployment recipe uses the same predeclared inner search on all development rows.
    # Outer scores are not an input to this selection function.
    final_scores = search(np.arange(len(y)), "final", SEED+900)
    choice = select_candidate(final_scores)
    focus = select_candidate(final_scores, focused=True)
    write_json(OUT / "deployment_selection.json", {"macro": choice, "recall": focus,
                "selection_input": "final inner cross-validation only; no outer scores or historic test scores"})
    roles = role_split(np.arange(len(y)), np.array([], dtype=int), "deployment", SEED+1000)
    for label, selected in [("macro", choice), ("recall", focus)]:
        bundle = fit_model(x, y, roles, by_name[selected["name"]], SEED)
        rows, _, rules, cal = score_bundle(bundle, x, y, roles, roles["stopping"])
        # Stopping scores here are not reported as generalization results.
        save_bundle(bundle, cal, OUT / f"models/{label}")
        write_json(OUT / f"models/{label}/decision.json", {"mode": selected["mode"], **rules[selected["mode"]],
                    "features": FEATURES, "kind": bundle["kind"], "name": selected["name"], "development_only": True})
    np.savez_compressed(OUT / "cache/role_row_ids.npz", **role_manifests)
    write_json(OUT / "role_audit.json", {"partitions": len(role_manifests), "all_roles_disjoint_from_evaluation": True,
               "historical_test_rows_used": 0, "grouped_outer_profile_overlap": 0,
               "row_id_manifest": "cache/role_row_ids.npz (local intermediate artifact)"})
    assert all(digest(ROOT / p) == h for p, h in historical_hashes.items()), "Historical artifacts were modified"
    write_json(OUT / "summary.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time()-start, "outer_folds": 3, "inner_folds": 2,
        "candidate_configurations": len(configs), "candidate_decision_variants": len(configs)*len(MODES),
        "historic_artifacts_unchanged": True, "historic_test_rows_used": 0, "development_rows": len(y),
        "final_model": choice, "recall_model": focus})
    print(f"Completed v2 in {(time.time()-start)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
