# Experiment 2: did the additional techniques help?

**The classification gains were small. Calibration improved probability-based scores. Prediabetes remained difficult.** The stricter evaluation also revealed how unstable some minority-class results are.

Experiment 2 used **228,312 development rows**, three outer folds and two inner folds. It excluded the historical test set. Read [How We Evaluate](How-We-Evaluate) for the fold diagram.

## What was added?

| Addition | What it tests |
|---|---|
| 18 training configurations, 108 score/decision variants | A broader search across MLPs, CatBoost settings and prediction rules |
| Three outer folds and two inner folds | Whether the whole model-selection procedure works on held-aside folds |
| Three neural training seeds | Whether stochastic training changes the result |
| Balanced sampling, random oversampling and SMOTENC | Whether changing training examples helps rare-class learning |
| Independent calibration | Whether class scores can better match observed outcomes |
| A recall-focused rule with a 5% precision floor | Whether more prediabetes cases can be found with a specified experimental tradeoff |
| Profile-grouped folds and subgroup analysis | Whether results vary with profile overlap or population subsets |

## The main classification result

The following are **means across the three outer folds**. They compare models within Experiment 2, not against Experiment 1's test scores.

| Procedure | Macro F1 | Prediabetes recall | Prediabetes precision |
|---|---:|---:|---:|
| Fixed weighted-MLP recipe | 0.4621 | 4.2% | 3.9% |
| Tuned MLP selection | 0.4633 | 10.4% | 4.2% |
| Tuned CatBoost selection | 0.4651 | 5.0% | 4.5% |
| Overall model-selection procedure | 0.4648 | 7.5% | 4.3% |

![Experiment 2 mean macro F1: 0.4621 fixed weighted MLP, 0.4633 tuned MLP, 0.4651 tuned CatBoost and 0.4648 overall selection.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/experiment-2-comparison.png)

Using the displayed values, the overall procedure improved macro F1 by **about 0.0027**. This is a modest gain, not a large breakthrough.

The tuned MLP found more prediabetes responses on average, but its 4.2% precision means most prediabetes predictions were still wrong. We cannot judge it by recall alone.

## Why can the CatBoost column be slightly better than “overall selection”?

The overall procedure chose using **inner-fold** scores. On the third outer fold, it chose an MLP. The separate CatBoost procedure then happened to score slightly higher on that outer evaluation. We did not change the choice after seeing those outcomes.

“Overall selection” describes the predeclared selection process, not a guarantee of the best possible outer score in every fold.

## Calibration helped probability-based scores

For paired comparisons of the same outer-refitted models:

| Probability metric | Before calibration | After calibration |
|---|---:|---:|
| Mean log loss | 0.4807 | **0.3945** |
| Mean multiclass Brier score | 0.2656 | **0.2212** |

Lower is better for these metrics. Calibration used its own training portion, separate from evaluation. These averages cover the paired refitted configurations in the calibration table; they are not the score of one final saved model.

Calibration and classification are different goals. Improving the scores' probabilistic behavior does not automatically improve recall or F1 under ordinary argmax. Both log loss and Brier score also reflect discrimination, so we also inspect reliability plots.

## The recall-focused policy did not succeed

Its goal was to maximize prediabetes recall while reaching at least 5% precision and macro F1 of 0.40 during inner selection. **The 5% precision floor was missed in all three outer folds.**

The outer-fold averages were approximately **3.9% prediabetes recall**, **4.2% precision** and **0.4226 macro F1**. Those results do not support replacing the macro-F1 procedure with this policy. The 5% setting is an experimental criterion, not a recommended clinical standard.

## Sampling was not a dependable improvement

Balanced replacement sampling, repeating prediabetes rows and creating SMOTENC samples did not consistently beat the reference approaches in the inner-validation comparisons. Synthetic or repeated data do not automatically add the missing information needed to distinguish a rare class.

The three sampling variants shared their architecture and regularization. The fixed reference recipes had different regularization or epoch budgets, so this comparison cannot isolate sampling as the sole cause of differences.

## What did seed and profile checks reveal?

- Across the three seeds, the selected MLP's mean prediabetes recall varied from about **5.0% to 10.4%**, while macro F1 varied much less. Rare-class behavior was sensitive to training randomness.
- A separate fixed-recipe grouped analysis kept matching input profiles out of both sides of each outer split. Mean macro F1 was about **0.4639**. Profiles are not person identities.
- There were **1,554 development input profiles with multiple different target labels**, covering 5,040 rows. Matching coarse survey answers do not always determine the same label; this is ambiguity, not proof of an annotation mistake.
- Subgroup tables report coded sex, age and income groups with support and prevalence. They are descriptive checks, not a completed fairness audit.

## Which final model was saved?

After the outer study, a separate inner search over all development rows selected the saved macro-F1 recipe: **`catboost_0`**, a depth-4 CatBoost model without class weighting. Decision tuning retained zero offsets for this final fit, so its saved rule uses ordinary argmax.

A separate focal-loss MLP was saved for the recall-focused research policy. The final inner-search scores are **selection scores**, not a new final test result. Neither saved artifact should be assumed clinically reliable.

**Evidence:** [fold results](https://github.com/NewOsakA/diabetes-ml/blob/experiment/robust-prediabetes/experiments/v2/outer_results.csv) · [technical report](https://github.com/NewOsakA/diabetes-ml/blob/experiment/robust-prediabetes/experiments/v2/REPORT.md) · [saved model decisions](https://github.com/NewOsakA/diabetes-ml/tree/experiment/robust-prediabetes/experiments/v2/models)

**Next:** [Run the project yourself →](Quick-Start)
