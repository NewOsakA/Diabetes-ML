"""Build the readable report and executed-notebook source from persisted results."""
import json

import nbformat as nbf
import pandas as pd

from .common import ROOT


def md_table(frame, digits=4):
    frame = frame.copy()
    for c in frame.select_dtypes(include="floating").columns:
        frame[c] = frame[c].map(lambda v: f"{v:.{digits}f}")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def build():
    results = ROOT / "results"
    table = pd.read_csv(results / "test_results.csv")
    experiments = pd.read_csv(results / "experiment_results.csv")
    audit = json.loads((results / "data_audit.json").read_text())
    manifest = json.loads((ROOT / "models/selection_manifest.json").read_text())
    registry = json.loads((ROOT / "models/registry.json").read_text())
    summary = json.loads((results / "run_summary.json").read_text())
    bootstrap = json.loads((results / "bootstrap_vs_mlp_baseline.json").read_text())
    baseline = table.set_index("model").loc["mlp_baseline"]
    champion = table.set_index("model").loc["champion"]
    delta = champion.macro_f1 - baseline.macro_f1
    classes = pd.read_csv(results / "champion_test_classes.csv")
    sensitivity = pd.read_csv(results / "profile_sensitivity.csv")
    selected = manifest["champion"]
    spec = registry[selected["model"]]
    top = experiments[experiments.split == "selection"].sort_values(["macro_f1", "balanced_accuracy"], ascending=False, kind="stable").head(10)
    comparison = table[~table.model.isin(["best_improved_mlp", "champion_argmax_ablation"])].copy()
    report = f'''# Diabetes ML — experiment report

Completed 28 October 2026. The validation-selected model improved held-out test macro F1 from **{baseline.macro_f1:.4f} to {champion.macro_f1:.4f}**: **+{delta:.4f} absolute**, or **{delta/baseline.macro_f1:.1%} relative**. Diabetes recall increased from **{baseline.diabetes_recall:.1%} to {champion.diabetes_recall:.1%}**. Accuracy decreased from **{baseline.accuracy:.1%} to {champion.accuracy:.1%}**. These are paired comparisons on the same test rows.

**Prediabetes remains poorly identified:** recall {champion.prediabetes_recall:.1%}, precision {champion.prediabetes_precision:.1%}. The model identifies only 29 of 463 prediabetes responses, and only 29 of 656 predictions of prediabetes are correct. The improvement primarily reflects better diabetes detection; this experiment does not solve prediabetes screening.

## Research question

How do imbalance-aware training and class-specific decision rules affect diabetes status classification from health and lifestyle survey indicators? This study compares logistic regression, multilayer perceptrons, gradient-boosted trees and soft-voting ensembles using identical training, validation and test partitions. Macro F1 is the primary model-selection metric, with class-level precision and recall used to assess minority-class performance.

## Data and safeguards

- Source: [Alex Teboul's Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), three-class BRFSS 2015 file.
- {audit['rows']:,} responses, {audit['feature_count']} predictors, zero missing cells. Class counts: no diabetes {audit['class_counts']['0']:,}; prediabetes {audit['class_counts']['1']:,}; diabetes {audit['class_counts']['2']:,}.
- Stratified seed-42 split: training 202,944; validation tuning 12,684; validation selection 12,684; test 25,368. Scaling and class weights come from training only.
- Fit checkpoints, boosting iterations, decision offsets and ensemble membership use tuning. Final candidate ranking uses selection macro F1. The timestamped manifest was saved before test evaluation.
- {summary['candidate_model_count']} fitted models/blends and {summary['decision_variants']} model/decision candidates. All models, histories, configurations and source checksums are saved.
- Preserve the original rows: {audit['duplicate_full_rows_after_first']:,} duplicate full rows after the first and {audit['duplicate_feature_profiles_after_first']:,} duplicate feature profiles. Survey answers can coincide for different people; no person IDs are available.

Source CSV SHA-256: `{audit['source_sha256']}`.

![Class distribution](figures/class_distribution.png)

## Selected model and evaluated techniques

The selected model is **{selected['model']}**, a ReLU network with hidden widths {spec['config']['hidden']}. It uses square-root inverse-frequency class weighting, dropout {spec['config'].get('dropout', 0)}, AdamW learning rate {spec['config']['lr']}, weight decay {spec['config'].get('weight_decay', 0)}, batch size 256, and 30 training epochs. The best checkpoint was epoch **{spec['config']['best_epoch']}**, selected on tuning macro F1.

The saved decision rule adds offsets `{selected['offsets']}` to log raw scores and takes argmax. This gives prediabetes a greater chance of being selected without changing the underlying ranking scores. Raw scores are not calibrated probabilities. Class weights actually used: `{[round(v, 4) for v in spec['config']['class_weights']]}`.

We also tested focal loss (gamma 2), fully balanced neural loss, histogram gradient boosting, CatBoost depth/weight variants, and tree/neural soft-voting ensembles. Greater model complexity did not consistently yield better selection scores. The validation-selected weighted MLP remained the final model.

![Training history](figures/mlp_sqrt_weighted_loss.png)

### Selection ranking (before opening the test results)

{md_table(top[['model', 'decision', 'macro_f1', 'balanced_accuracy', 'prediabetes_recall']])}

Selection macro F1 for the champion was **{selected['selection_macro_f1']:.4f}**. The boosting finalist achieved a slightly higher test macro F1 ({table.set_index('model').loc['best_boosted_tree','macro_f1']:.4f}) than the chosen model; the model was **not replaced after test inspection**.

## Held-out test results

{md_table(comparison[['model', 'accuracy', 'balanced_accuracy', 'macro_f1', 'prediabetes_recall', 'diabetes_recall']])}

`best_improved_mlp` and `champion` refer to the same selected model, so the duplicate is omitted from this table. The full CSV retains all pre-frozen comparisons.

![Test comparison](figures/test_comparison.png)

### Per-class results for the selected model

{md_table(classes)}

Specificity is one-versus-rest. Average precision uses raw one-versus-rest ranking scores. Precision and recall use the final decision rule. Class-1 average precision is low, consistent with the difficult prediabetes distinction.

![Test confusion matrix](figures/champion_test_confusion.png)

![Precision–recall curves](figures/champion_precision_recall.png)

### Where the improvement comes from

{md_table(table[table.model.isin(['mlp_baseline','champion_argmax_ablation','champion'])][['model','accuracy','macro_f1','balanced_accuracy','prediabetes_recall','diabetes_recall']])}

Weighted training/regularization raises macro F1 even before decision tuning. Decision tuning then adds minority-class predictions and improves macro F1 further. This is an empirical comparison: architecture regularization and class weights were changed together, so the experiment does not isolate each training ingredient causally.

### Uncertainty

The paired stratified bootstrap (2,000 resamples) gives a **95% interval of [{bootstrap['macro_f1_delta_95_ci'][0]:.4f}, {bootstrap['macro_f1_delta_95_ci'][1]:.4f}]** for macro-F1 improvement over our MLP baseline. The champion macro-F1 interval is [{bootstrap['improved_macro_f1_95_ci'][0]:.4f}, {bootstrap['improved_macro_f1_95_ci'][1]:.4f}].

These intervals condition on one partition and trained models. They do not include retraining variation, model selection uncertainty, profile clustering or survey-design effects.

### Duplicate-profile sensitivity

{md_table(sensitivity[['model','subset','rows','accuracy','macro_f1','prediabetes_recall','diabetes_recall']])}

The champion's macro-F1 improvement also appears among the 21,888 test rows with feature profiles absent from training. This is a descriptive subset check; it is not a separate group-held-out study. Repeated profiles have different class composition, so compare models within each subset.

## Remaining limits and follow-up research

Prediabetes sensitivity and precision remain low. Fully balanced models showed higher minority recall on selection but lower macro F1/accuracy; see the experiment CSV before choosing a different operating objective. No further tuning was performed after test results were available. A future recall-constrained experiment should use a new evaluation protocol or nested cross-validation, rather than repeatedly optimizing this test set.

This is cross-sectional status classification using self-reported 2015 US survey responses, not prediction of future onset. Model associations do not establish causes. No survey weights, prospective validation, subgroup fairness study or clinical calibration were performed. The results support a research analysis of this dataset; clinical use requires further validation.

## Project files

- `../README.md`: setup and runnable commands.
- `../METHODOLOGY.md`: full experiment specification and technical references.
- `../DATA_DICTIONARY.md`: feature meanings and valid survey codes.
- `../notebooks/01_exploration_and_results.ipynb`: executed exploration and results notebook.
- `experiment_results.csv`: tuning/selection outcomes for all candidates.
- `test_results.csv`: frozen held-out comparisons.
- `../models/selection_manifest.json`: selection timestamp, exact model and decision rule.
- `histories/`, `figures/`, `training.log`: training evidence and presentation artifacts.
'''
    (results / "REPORT.md").write_text(report)

    # The notebook reads saved evidence; it does not retrain or tune on test data.
    notebook = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook.cells = [
        md("# Diabetes ML: imbalance-aware status classification\n\nInvestigate how class weighting, focal loss, boosted trees and decision tuning affect three-class diabetes status classification from BRFSS health indicators. This notebook presents data exploration and the evaluated model comparison. It reads saved artifacts; training is a separate, explicit command. See `METHODOLOGY.md` for the experimental protocol."),
        code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image, Markdown\nfrom diabetes_ml.common import ROOT, FEATURES, TARGET\nraw = pd.read_csv(ROOT / 'data/raw/diabetes_012_health_indicators_BRFSS2015.csv')\nprint('Project:', ROOT)\nprint('Shape:', raw.shape)\ndisplay(raw.head())"),
        md("## Schema and quality\nZero is a valid survey response, not a missing-value code. Ordinal fields, including Age, are categories rather than literal years."),
        code("audit = json.loads((ROOT / 'results/data_audit.json').read_text())\ndisplay(pd.Series({k: audit[k] for k in ['rows','feature_count','missing_cells','duplicate_full_rows_after_first','duplicate_feature_profiles_after_first','test_profiles_seen_in_training']}))\ndisplay(raw.describe().T)"),
        md("## Class imbalance and feature distributions\nThe prediabetes class contains only about 1.8% of responses."),
        code("display(raw[TARGET].value_counts().sort_index().rename_axis('class').to_frame('count'))\ndisplay(Image(filename=str(ROOT / 'results/figures/class_distribution.png')))\ndisplay(Image(filename=str(ROOT / 'results/figures/indicator_distributions.png')))"),
        code("display(Image(filename=str(ROOT / 'results/figures/feature_correlations.png')))"),
        md("## Fixed partitions and training-only preprocessing\n80% training, 5% tuning, 5% selection, 10% test. Persisted row IDs are never predictors. Selection chooses the final model before test inference."),
        code("display(pd.read_csv(ROOT / 'results/split_counts.csv'))\nmanifest = json.loads((ROOT / 'models/selection_manifest.json').read_text())\ndisplay(manifest['champion'])\nprint('Frozen before test evaluation:', manifest['frozen_at_utc'])"),
        md("## Candidate comparison on selection data\nFocal loss, class weighting, boosted trees, decision offsets and soft voting are compared using macro F1. Selection results are model-selection estimates, not final generalization claims."),
        code("experiments = pd.read_csv(ROOT / 'results/experiment_results.csv')\nselection = experiments[experiments.split.eq('selection')].sort_values(['macro_f1','balanced_accuracy'], ascending=False, kind='stable')\ndisplay(selection[['model','decision','accuracy','balanced_accuracy','macro_f1','prediabetes_recall']].reset_index(drop=True))"),
        md("## Training curves\nThe baseline uses ordinary cross-entropy. Other train objectives may be weighted or focal, while validation CE is always ordinary cross-entropy; their loss scales differ."),
        code("for name in ['mlp_baseline','mlp_sqrt_weighted','mlp_balanced','mlp_focal']:\n    display(Image(filename=str(ROOT / f'results/figures/{name}_loss.png')))"),
        md("## Logistic coefficient associations\nInputs are standardized using training statistics. Coefficients describe model associations, not causes."),
        code("display(Image(filename=str(ROOT / 'results/figures/logistic_coefficients.png')))"),
        md("## Frozen held-out test comparison\nThe validation-selected champion is retained even if another frozen model has a slightly higher test score. Best improved MLP and champion are the same fitted model."),
        code("test_results = pd.read_csv(ROOT / 'results/test_results.csv')\ndisplay(test_results)\ndisplay(Image(filename=str(ROOT / 'results/figures/test_comparison.png')))"),
        code("display(pd.read_csv(ROOT / 'results/champion_test_classes.csv'))\ndisplay(Image(filename=str(ROOT / 'results/figures/champion_test_confusion.png')))\ndisplay(Image(filename=str(ROOT / 'results/figures/champion_precision_recall.png')))"),
        md("## Confidence intervals and profile sensitivity\nPaired bootstrap intervals condition on these trained models. Unseen-profile test results are descriptive and are not a second independent evaluation."),
        code("display(json.loads((ROOT / 'results/bootstrap_vs_mlp_baseline.json').read_text()))\ndisplay(pd.read_csv(ROOT / 'results/profile_sensitivity.csv'))"),
        md("## Saved-model inference example\nRaw scores are not calibrated clinical probabilities. Prediabetes precision and recall remain poor; do not interpret these predictions as diagnoses."),
        code("from diabetes_ml.predict import Predictor\nexample = pd.read_csv(ROOT / 'data/processed/example_inputs.csv')\ndisplay(Predictor().predict(example))"),
        md("## Interpretation\nThe selected weighted MLP improves macro F1 and diabetes recall at a cost to accuracy. Prediabetes remains difficult. The full report states the measured tradeoffs and single-split limitations. No further optimization should use this exposed test set."),
    ]
    notebook.metadata = {"kernelspec": {"display_name": "Diabetes ML (.venv)", "language": "python", "name": "diabetes-ml"},
                         "language_info": {"name": "python", "version": "3.12"}}
    nbf.write(notebook, ROOT / "notebooks/01_exploration_and_results.ipynb")
    splits = np_load_splits()
    raw = pd.read_csv(ROOT / "data/raw/diabetes_012_health_indicators_BRFSS2015.csv")
    # Existing held-out rows provide a saved-model inference example.
    sample = raw.iloc[splits["test"][:5]][[c for c in raw.columns if c != "Diabetes_012"]]
    sample.to_csv(ROOT / "data/processed/example_inputs.csv", index=False)


def np_load_splits():
    import numpy as np
    return np.load(ROOT / "data/processed/splits.npz")


if __name__ == "__main__":
    build()
