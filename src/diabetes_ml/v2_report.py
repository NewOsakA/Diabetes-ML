"""Create v2 scientific figures, report and analysis notebook from saved evidence."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay

from .build_report import md_table
from .common import ROOT, CLASSES, FEATURES
from .v2_protocol import OUT, load_development


def build():
    development, _, _ = load_development()
    development[FEATURES].head().to_csv(OUT / "example_inputs.csv", index=False)
    summary = json.loads((OUT / "summary.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    profile = json.loads((OUT / "profile_audit.json").read_text())
    oof_table = pd.read_csv(OUT / "oof_summary.csv")
    outer = pd.read_csv(OUT / "outer_results.csv")
    seeds = pd.read_csv(OUT / "seed_results.csv")
    calibration = pd.read_csv(OUT / "calibration_results.csv")
    subgroups = pd.read_csv(OUT / "subgroup_results.csv")
    grouped = pd.read_csv(OUT / "grouped_sensitivity.csv")
    inner = pd.read_csv(OUT / "inner_results.csv")
    data = np.load(OUT / "cache/oof_partial.npz")
    final_inner = inner[inner.search == "final"].groupby(["name", "kind", "mode"], as_index=False).mean(numeric_only=True)
    (OUT / "figures").mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    def save(name):
        plt.tight_layout()
        plt.savefig(OUT / f"figures/{name}.png", dpi=170, bbox_inches="tight")
        plt.close()
    names = {"fixed_plain_mlp": "Plain MLP", "fixed_weighted_mlp": "Fixed weighted MLP",
             "selected_mlp": "Tuned MLP", "selected_catboost": "Tuned CatBoost",
             "prediabetes_focused": "Recall-focused", "selected_overall": "Selected pipeline"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["macro_f1", "prediabetes_recall", "prediabetes_precision"]):
        for j, method in enumerate(names):
            values = outer.loc[outer.method == method, metric]
            ax.barh(j, values.mean(), xerr=values.std(), color="#527c96", alpha=.8, capsize=3)
            ax.scatter(values, np.full(len(values), j), color="#162b42", s=20, zorder=5)
        ax.set_yticks(range(len(names)), list(names.values()))
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_xlim(left=0)
    fig.suptitle("Development-only nested CV: dots are outer folds; error bars show fold SD")
    save("nested_comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ["fixed_plain_mlp", "selected_mlp", "selected_catboost", "prediabetes_focused"]:
        PrecisionRecallDisplay.from_predictions(data["y"] == 1, data[f"{method}_proba"][:, 1], name=names[method], ax=ax)
    ax.axhline((data["y"] == 1).mean(), linestyle="--", color="gray", label="Prediabetes prevalence")
    ax.set(title="Prediabetes ranking (precision axis zoomed to 0–0.25)", ylim=(0, .25))
    ax.legend(fontsize=9)
    save("prediabetes_precision_recall")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, method in zip(axes, ["selected_overall", "prediabetes_focused"]):
        cm = pd.read_csv(OUT / f"{method}_confusion.csv").to_numpy()
        normalized = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(normalized, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False)
        ax.set(title=names[method], xlabel="Predicted class", ylabel="True class")
        ax.tick_params(axis="y", rotation=0)
    save("oof_confusion")
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in ["selected_overall", "prediabetes_focused"]:
        observed, predicted = calibration_curve(data["y"] == 1, data[f"{method}_proba"][:, 1], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=names[method])
    ax.plot([0, .2], [0, .2], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted prediabetes score", ylabel="Observed prediabetes fraction", title="Prediabetes reliability: pooled out-of-fold scores")
    ax.legend()
    save("reliability")

    average = outer.groupby("method").agg({c: ["mean", "std"] for c in ["macro_f1", "prediabetes_recall", "prediabetes_precision", "prediabetes_ap"]})
    display = pd.DataFrame({"method": average.index})
    for metric in ["macro_f1", "prediabetes_recall", "prediabetes_precision", "prediabetes_ap"]:
        display[metric+" mean ± SD"] = [f"{mean:.4f} ± {sd:.4f}" for mean, sd in zip(average[(metric,"mean")], average[(metric,"std")])]
    calibration_means = calibration.groupby("mode")[["log_loss", "brier", "prediabetes_ap"]].mean().reset_index()
    fixed = oof_table.set_index("method").loc["fixed_weighted_mlp"]
    selected = oof_table.set_index("method").loc["selected_overall"]
    focused = oof_table.set_index("method").loc["prediabetes_focused"]
    recall_floor_folds = outer[(outer.method == "prediabetes_focused") & (outer.prediabetes_precision >= protocol["precision_floor"])]
    # Sampling ablations use paired inner-validation evaluations; they are not outer estimates.
    sampling = inner[inner.search.str.startswith("outer") & inner["mode"].eq("raw_macro") &
                     inner.name.isin(["fixed_plain_mlp", "fixed_weighted_mlp", "mlp_balanced_sampling", "mlp_random_oversampling", "mlp_smotenc"])]
    sampling_summary = sampling.groupby("name")[["macro_f1", "prediabetes_recall", "prediabetes_precision", "prediabetes_ap"]].mean().reset_index()
    sampling_summary.to_csv(OUT / "sampling_inner_summary.csv", index=False)
    seed_summary = seeds.groupby("seed")[["macro_f1", "prediabetes_recall", "prediabetes_precision"]].agg(["mean","std"])
    seed_summary.to_csv(OUT / "seed_summary.csv")
    selected_groups = subgroups[subgroups.method.eq("selected_overall")]
    group_means = grouped[["macro_f1", "prediabetes_recall", "prediabetes_precision"]].mean()
    report = f'''# Diabetes ML — development experiment v2

The new experiment evaluates tuning, sampling and calibration with **3 outer folds × 2 inner folds**, using only the original **228,312 development rows**. Historical test rows used: **0**. Original models, results and split IDs remain byte-for-byte unchanged.

The selected procedure obtained pooled out-of-fold macro F1 **{selected.macro_f1:.4f}**, compared with **{fixed.macro_f1:.4f}** for the fixed weighted-MLP recipe re-fitted in the same outer folds. The recall-focused procedure obtained prediabetes recall **{focused.prediabetes_recall:.1%}** at precision **{focused.prediabetes_precision:.1%}**, with macro F1 **{focused.macro_f1:.4f}**. These are development-stage comparisons; they are not directly interchangeable with the original held-out test scores.

## Protocol and interpretation

- Outer folds evaluate the entire inner-search procedure. Each inner search compares 18 training configurations and six score/decision modes (108 candidates).
- Within each inner or outer training partition: 70% fit model/scaler/sampler; 10% choose training checkpoint; 10% fit a positive per-class log-probability scaling-and-bias calibrator; 10% tune decisions. All roles are disjoint from evaluation rows.
- Inner-validation mean macro F1 selects MLP, CatBoost and overall procedures independently. A separate procedure maximizes prediabetes recall subject to mean inner precision ≥5% and macro F1 ≥0.40. The 5% threshold and minimum 30 decision-set positive predictions are experimental safeguards, not clinical targets.
- Decision rules are tuned before outer evaluation. The precision requirement is **not guaranteed on new rows**; the focused procedure achieved ≥5% precision in **{len(recall_floor_folds)} of 3 outer folds**. Infeasible searches explicitly fall back to the macro-F1 procedure.
- Training seeds 42, 43 and 44 assess stochastic variation in the chosen MLP with unchanged outer partitions. Seed results do not influence model selection.
- These methods were motivated by earlier exploration of this dataset. Nested CV estimates this predeclared development procedure; an independent external dataset is still needed for a new final generalization claim.

## Outer-fold results

{md_table(display)}

The ± values are standard deviations across three outer folds, not confidence intervals. Overlapping training folds and only three evaluations limit uncertainty claims.

The broader search should be judged against the fixed weighted recipe in these same folds. Its classification gains are modest. The recall-focused procedure does not provide a dependable improvement: its precision target can fail outside the decision and inner-validation portions, and its macro F1 is lower. Additional complexity alone has not resolved the rare-class problem.

![Nested comparisons](figures/nested_comparison.png)

### Pooled out-of-fold results

{md_table(oof_table)}

Each response is scored once by a pipeline that excluded its outer fold from fitting, calibration, decision tuning and hyperparameter selection. Different outer folds may choose different configurations. This evaluates a **selection procedure**, not one globally fitted model. The full fold-specific choices are saved in `fold*_frozen_choices.json`.

Pooled precision–recall curves combine scores from different fold-specific models, which may have different score scales. Use mean outer-fold average precision as the primary comparison of ranking quality; pooled curves are descriptive.

![Confusion matrices](figures/oof_confusion.png)

![Prediabetes precision–recall](figures/prediabetes_precision_recall.png)

## Calibration

The custom calibrator fits five parameters to an independent calibration portion: three positive scales applied to log probabilities and two class biases (class-0 bias fixed to zero). Optimization minimizes multiclass log loss with mild parameter regularization. This is distinct from the existing decision offsets.

The following pairs evaluate the same fitted models on the same outer rows, before and after calibration. Means average the distinct outer-refitted configurations; this table is an ablation, not a selection criterion.

{md_table(calibration_means)}

Lower log loss and multiclass Brier score indicate better probabilistic performance overall; neither alone isolates calibration from discrimination. Vector scaling may also change ranking. Reliability plots help inspect calibration, especially for the rare class.

![Reliability](figures/reliability.png)

## Sampling experiments

{md_table(sampling_summary)}

These are **inner-validation averages**, included to compare training strategies. They are not additional outer test estimates. Random oversampling and SMOTENC increase prediabetes training count to 15% of the no-diabetes count; they do not balance the evaluation distribution. Class-balanced replacement sampling draws equal expected class frequencies per epoch. Sampling variants use unweighted loss to avoid double correction.

The random-oversampling, SMOTENC and balanced-sampling configurations share their architecture, regularization and epoch budget. The fixed reference recipes have different regularization or epoch settings, so differences against those references cannot be attributed to sampling alone.

SMOTENC treats binary and ordinal survey answers as categorical; BMI and mental/physical unhealthy-day counts are numeric. Categorical codes are asserted valid after generation. Synthetic numeric combinations are still approximate survey profiles and are not new observed respondents. Scaling is fitted on the original fit rows; samplers never see calibration, decision or evaluation rows.

## Seed robustness

{md_table(seeds[['fold','seed','name','mode','macro_f1','prediabetes_recall','prediabetes_precision']])}

Only the training seed changes within each outer fold. Variation combines stochastic model fitting with recalibration and decision tuning on fixed row portions. No best seed is chosen from these outer scores.

## Profile and subgroup analysis

The development set contains **{profile['unique_feature_profiles']:,} unique feature profiles**, including **{profile['profiles_with_multiple_labels']:,} profiles with more than one target label**. These cover **{profile['rows_in_conflicting_profiles']:,} rows**, including **{profile['prediabetes_rows_in_conflicting_profiles']:,} prediabetes rows. Coarse answers can match for distinct people, so conflicting labels are not automatically labeling mistakes. They indicate ambiguity in the available features.

A separate three-fold feature-profile-grouped sensitivity run uses the **fixed weighted-MLP recipe**, with no hyperparameter selection on grouped evaluation folds. No feature profile crosses a grouped training/evaluation boundary. Mean macro F1: **{group_means.macro_f1:.4f}**; prediabetes recall **{group_means.prediabetes_recall:.1%}**; precision **{group_means.prediabetes_precision:.1%}**. This tests an unseen-profile setting; profiles are not verified person identities.

{md_table(selected_groups[['subgroup','rows','prediabetes_support','prediabetes_prevalence','macro_f1','prediabetes_recall','prediabetes_precision','prediabetes_ap']])}

Subgroup differences are descriptive and reflect sample sizes, prevalence and case mix. They do not establish discrimination or a completed fairness audit. Prediabetes support and prevalence are shown to make these comparisons interpretable.

## Saved development models

Final macro-F1 recipe: **{summary['final_model']['name']} / {summary['final_model']['mode']}**. Final recall-focused recipe: **{summary['recall_model']['name']} / {summary['recall_model']['mode']}**. Selection uses a separate two-fold inner search over all development rows, without using outer scores as an input. The same fit/stopping/calibration/decision partitions are retained for the deployment fits.

Final recall-search constraint feasible on mean inner-validation results: **{summary['recall_model']['selection_constraint_feasible']}**. A false value indicates the explicitly documented fallback to the macro-F1 choice, rather than a model satisfying the precision target.

These saved models are research artifacts. Their final inner-search scores are model-selection scores, not fresh held-out performance. Original inference remains available via `diabetes_ml.predict`; v2 inference uses `diabetes_ml.v2_predict --policy macro` or `--policy recall`.

## Files and execution

- `protocol.json`: exact candidates, data/code hashes, package versions and role proportions.
- `inner_results.csv`, `outer_results.csv`, `outer_summary.csv`, `oof_summary.csv`: selection and evaluation evidence.
- `seed_results.csv`, `calibration_results.csv`, `grouped_sensitivity.csv`, `subgroup_results.csv`: diagnostic experiments.
- `models/macro/`, `models/recall/`: fitted model, scaler if needed, calibration parameters and frozen decision rule.
- `role_audit.json`, `historical_hashes.json`: partition checks and preservation evidence.
- `cache/`: local row-ID manifests, complete OOF scores and restart caches; excluded from Git.
- `training.log`, `figures/`, `analysis.ipynb`: readable evidence.

Completed in {summary['elapsed_seconds']/60:.1f} minutes on the recorded local environment. Run from the repository root:

```bash
.venv/bin/python -m pip install -e '.[dev,experiments]'
.venv/bin/python -u -m diabetes_ml.v2_run > experiments/v2/training.log 2>&1
.venv/bin/python -m diabetes_ml.v2_report
.venv/bin/python -m diabetes_ml.v2_predict --input experiments/v2/example_inputs.csv --output experiments/v2/example_predictions.csv
```

Completed results are protected against overwrite. `--resume` reuses inner results only when the saved protocol and source hashes match exactly; it refits outer and final models. To change the protocol after completion, use a separate experiment directory rather than replacing this evidence.

## Method references

- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
- [Stratified group folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)
'''
    (OUT / "REPORT.md").write_text(report)
    notebook = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook.cells = [
        md("# Diabetes ML v2: nested evaluation and prediabetes experiments\n\nDevelopment-only analysis. Original test results remain historical. See REPORT.md for the protocol and limits."),
        code("import json\nimport pandas as pd\nfrom IPython.display import display, Image\nfrom diabetes_ml.v2_protocol import OUT\ndisplay(json.loads((OUT/'summary.json').read_text()))\ndisplay(pd.read_csv(OUT/'oof_summary.csv'))"),
        md("## Outer-fold variability\nDots show separate outer evaluations; error bars are fold standard deviations."),
        code("display(pd.read_csv(OUT/'outer_results.csv'))\ndisplay(Image(filename=str(OUT/'figures/nested_comparison.png')))\ndisplay(Image(filename=str(OUT/'figures/oof_confusion.png')))"),
        md("## Prediabetes discrimination and calibration"),
        code("display(Image(filename=str(OUT/'figures/prediabetes_precision_recall.png')))\ndisplay(pd.read_csv(OUT/'calibration_results.csv'))\ndisplay(Image(filename=str(OUT/'figures/reliability.png')))"),
        md("## Sampling, seeds, profile ambiguity and subgroups"),
        code("display(pd.read_csv(OUT/'sampling_inner_summary.csv'))\ndisplay(pd.read_csv(OUT/'seed_results.csv'))\ndisplay(json.loads((OUT/'profile_audit.json').read_text()))\ndisplay(pd.read_csv(OUT/'grouped_sensitivity.csv'))\ndisplay(pd.read_csv(OUT/'subgroup_results.csv'))"),
        md("## Saved development-model inference\nThese example predictions demonstrate artifact loading, not fresh validation."),
        code("from diabetes_ml.v2_predict import predict\ninputs=pd.read_csv(OUT/'example_inputs.csv')\ndisplay(predict(inputs, policy='macro'))\ndisplay(predict(inputs, policy='recall'))"),
    ]
    notebook.metadata = {"kernelspec": {"display_name": "Diabetes ML (.venv)", "language": "python", "name": "diabetes-ml"}}
    nbformat.write(notebook, OUT / "analysis.ipynb")


if __name__ == "__main__":
    build()
