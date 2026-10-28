# How do we check whether a model works?

Think of training as practice and evaluation as an exam. **A model should not receive the exam answers while it is being fitted or chosen.** Otherwise its score can look better than its real ability to generalize.

## Experiment 1: one fixed split

![Training uses 80% of rows, tuning 5%, selection 5% and the final test 10%.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/data-partitions.png)

| Portion | Rows | Job |
|---|---:|---|
| Training | 202,944 | Fit the scaler, class weights and model parameters |
| Tuning | 12,684 | Choose checkpoints, boosting iterations and decision offsets |
| Selection | 12,684 | Choose among the model/decision candidates |
| Test | 25,368 | Evaluate the frozen comparisons |

The split is **stratified**, meaning it approximately preserves the class proportions in each portion. A **random seed** fixes the randomized partition so the same setup can be checked again.

The model choice and decision settings were saved before calculating test performance. Another frozen model scored slightly higher on the test set, but we retained the validation-selected choice.

## Experiment 2: nested cross-validation

Experiment 2 uses only the **228,312 development rows** from Experiment 1's training and validation portions. The 25,368 historical test rows do not enter its fitting, tuning or evaluation.

![Diagram: How We Evaluate](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/nested-evaluation.png)

**Inner folds choose settings. Outer folds evaluate the selection procedure.** This is why the design is called “nested.” Each outer evaluation can select a different model configuration. [Nested cross-validation documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

Inside each training partition, data have four more distinct jobs:

| Role | Share of that training partition | Purpose |
|---|---:|---|
| Fit | 70% | Fit model/scaler/sampler |
| Stopping | 10% | Choose training checkpoint or boosting iteration |
| Calibration | 10% | Adjust score calibration |
| Decision | 10% | Choose the final label rule |

These percentages apply **inside a training partition**, not directly to the full 253,680-row dataset.

## What does “out of fold” mean?

Each development response receives an evaluation prediction from a pipeline that excluded its outer fold from fitting and selection. These predictions can be pooled into one confusion matrix.

Two useful summaries can differ slightly:

- **Mean outer-fold score:** calculate the metric in each fold, then average.
- **Pooled out-of-fold score:** join all evaluation predictions, then calculate the metric once.

F1 is nonlinear, so these need not be identical. The selected procedure's macro F1 is **0.4648 as a fold mean** and **0.4659 when pooled**. Both are valid labeled summaries of this run.

For ranking metrics such as AP, use the mean fold result as the primary comparison: pooled scores can mix different model score scales.

## Other checks

- **Three neural seeds:** repeat stochastic training with seeds 42, 43 and 44 on unchanged role assignments.
- **Profile-grouped folds:** keep identical input profiles on the same side of the training/evaluation boundary in a separate sensitivity experiment. Profiles are not verified person IDs.
- **Subgroup results:** inspect performance by coded sex, age and income groups, with support and prevalence shown.
- **Automated checks:** verify partition isolation, valid categories, calibration, metrics, saved-model loading and unchanged historical artifacts.

Three folds give limited evidence about uncertainty. Fold standard deviation is not a confidence interval. Also, these data had already been explored before Experiment 2; nested CV does not turn them into a completely new external dataset.

**Next:** [Experiment 1 findings →](Experiment-1-Results)
