# Experimental protocol

## Research objective

Investigate how imbalance-aware training and class-specific decision rules affect three-class diabetes status classification from health and lifestyle survey indicators. Compare logistic regression, multilayer perceptrons, gradient-boosted trees and soft-voting ensembles on identical data partitions. Prioritize macro F1 while reporting minority-class precision and recall, balanced accuracy and overall accuracy to make the tradeoffs explicit.

## Data and partitioning

Download the original Kaggle archive and extract only `diabetes_012_health_indicators_BRFSS2015.csv`. Record timestamps and SHA-256 hashes. Keep all original rows. Verify 21 inputs, target classes 0/1/2, finite data, no missing cells and expected survey-code ranges.

Use seed 42 with stratification: 80% training, 10% validation, 10% test. Split validation evenly into **tuning** (5% of all rows) and **selection** (5%). Tuning selects checkpoints, boosting iterations, ensemble members and decision offsets. Selection chooses the final fitted model and rule. It is still a model-selection set and its best score is optimistic; only the test result is the final evaluation. Persist original row positions, which are excluded from model inputs.

StandardScaler is fitted once using training data only, then reused by logistic regression and MLPs. Trees use raw numeric survey codes. No oversampling or synthetic respondents are used. Duplicate profiles are retained because coarse responses may coincide for distinct respondents. Report performance separately for test feature profiles seen and unseen in training. This sensitivity analysis is not an independent group-held-out retraining study and does not establish person-level independence.

## Predeclared candidate family

1. Majority-class dummy baseline.
2. Multiclass logistic regression: unweighted C=1; balanced class weights at C=0.1, 1, 10; lbfgs, max_iter=1500.
3. MLP baseline: 21→64→32→3, ReLU, 30 epochs, batch 256, AdamW with learning rate 0.001 and weight decay 0 (equivalent Adam update in this configuration).
4. Square-root inverse-frequency weighted MLP: 64→32, lr=0.001, dropout=0.1, decay=0.0001.
5. Fully inverse-frequency weighted MLP: 128→64, lr=0.0005, dropout=0.2, decay=0.001.
6. Focal-loss MLP: 128→64, lr=0.001, dropout=0.15, decay=0.0001, gamma=2, square-root inverse-frequency alpha.
7. Histogram gradient boosting: 300 maximum iterations, learning_rate=0.06, 15 leaves, L2=5; class-weight powers 0, 0.5, 1. Early stopping uses an internal 10% split of training data.
8. CatBoost: maximum 900 iterations, L2=5, numeric inputs, 4 CPU threads. Configurations (depth, class-weight power, learning rate): (5,0,0.06), (6,0.5,0.06), (7,0.5,0.04), (6,1,0.06). Early stopping patience 100 on unweighted tuning macro F1.
9. Two soft-voting ensembles: the two highest-ranked tree models on tuned tuning macro F1 with equal weights; a second blend gives these models 0.4 each and the highest-ranked MLP 0.2. Members are chosen on tuning, before selection or test ranking.

All MLPs train 30 epochs and retain the highest tuning macro-F1 checkpoint. CPU execution, four Torch threads, deterministic algorithms and explicit seeds reduce run variation. The training log and histories retain every epoch. Focal probability `pt` is computed from ordinary, unweighted CE, then alpha is applied; tests verify this detail.

Class weights use `(N / (3 * n_class)) ** power`, estimated from training labels only. Neural weights are normalized to have average training-example weight one. Weighted training can improve minority sensitivity at the cost of accuracy and calibration. Weighted/focal objectives are not numerically comparable with ordinary validation CE.

## Decision tuning and selection

Evaluate ordinary argmax and a cost-sensitive rule for each model. The tuned rule predicts `argmax(log(raw_score) + offset)`, fixing class-0 offset to zero. Grid: class-1 offset -1 to 4, class-2 offset -0.5 to 2, both in steps of 0.25. Include the unchanged argmax as a fallback. Choose offsets on tuning macro F1 only.

This is a custom multiclass decision rule inspired by the distinction between learning scores and choosing decisions; it is not scikit-learn's binary threshold tuner. Raw scores remain unmodified and are **not calibrated clinical probabilities**, especially for weighted/focal models and blends. Changing decisions does not improve the underlying ranking/AUPRC.

Rank candidates on selection macro F1, breaking ties with balanced accuracy. Save a timestamped selection manifest, exact offsets, ensemble weights and artifact references **before test inference**. Freeze baseline, weighted-logistic, improved-neural, boosted-tree and champion comparisons at the same time. An ordinary-argmax champion ablation isolates the effect of its decision rule. Do not revise the candidate family or tune after inspecting test results.

## Evaluation

Report accuracy, balanced accuracy (mean per-class recall), macro F1, prediabetes precision and recall, diabetes recall, and each class's precision, recall, specificity, F1, support and one-versus-rest average precision. Specificity is TN/(TN+FP) with each class treated as positive in turn. Undefined precision/F1 for absent predictions is zero. Matrices have true labels in rows and predicted labels in columns; show counts and row-normalized percentages.

The combined 10% validation plots summarize model behavior but are descriptive because their tuning half helped choose checkpoints and decisions. Use the separate selection scores for model ranking and held-out test scores for final claims. Compare models on the same rows for a paired assessment of improvement.

Use 2,000 paired stratified row-bootstrap replicates for the champion's macro-F1 difference against local MLP and logistic baselines. Preserve pairings by resampling joint true-class/baseline-prediction/champion-prediction cells. Intervals condition on this split and fitted models; they do not cover training-seed variation, model selection, profile clustering or the BRFSS survey design.

## Limits

This is a single-split research experiment on self-reported 2015 US survey data. Prediabetes is rare (about 1.8%) and hard to distinguish with these inputs. Fairness, prospective utility, probability calibration, external generalization and clinical deployment require further work. Better macro F1 does not imply every class or accuracy improved. Prediction is an educational inference demonstration, not a diagnosis.

## Technique references

- [CatBoost tuning and early stopping](https://catboost.ai/docs/en/concepts/parameter-tuning)
- [scikit-learn decision-threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Focal loss paper](https://arxiv.org/abs/1708.02002) — introduced for dense object detection; applied here experimentally to tabular multiclass classification.
- [Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
