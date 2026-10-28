# Experiment 1: what improved?

**The weighted neural network found more diabetes cases, while making more false-positive predictions.** Prediabetes remained poorly identified.

These results use the same **25,368 held-out test rows** for every frozen comparison. Settings were selected before test evaluation.

![The weighted MLP reduces accuracy from 84.8% to 80.5%, increases diabetes recall from 23.3% to 51.0%, and increases prediabetes recall from 0% to 6.3%.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/experiment-1-tradeoff.png)

## The main comparison

| Test metric | Baseline MLP | Selected weighted MLP | How to read it |
|---|---:|---:|---|
| Accuracy | 84.8% | 80.5% | Fewer correct labels overall |
| Macro F1 | 0.4128 | 0.4675 | Better combined precision/recall when classes count equally |
| Balanced accuracy | 40.0% | 48.1% | Better average class recall |
| Diabetes recall | 23.3% | 51.0% | Finds more actual diabetes cases |
| Prediabetes recall | 0% | 6.3% | A small improvement from zero, still poor |

Macro F1 increased by **0.0547 absolute**, or **13.2% relative**. The project did not improve every metric simultaneously.

## What changed in actual counts?

There were **3,534 actual diabetes responses** in the test set:

- Baseline MLP identified **822** correctly.
- Weighted MLP identified **1,801** correctly.
- That is **979 additional correct detections** on these rows.

But the weighted model made **4,282 diabetes predictions**, of which 1,801 were correct. Its diabetes precision was **42.1%**. The remaining predictions belonged to one of the other two classes.

For prediabetes, it identified **29 of 463 actual responses**. It predicted prediabetes **656 times**, giving precision **29 ÷ 656 = 4.4%**. A positive prediction for this class was usually incorrect.

See [Understanding Metrics](Understanding-Metrics) for the complete confusion matrix and worked calculations.

## What model was selected?

| Setting | Value |
|---|---|
| Model | MLP neural network |
| Layer widths | 21 → 64 → 32 → 3 |
| Hidden activation | ReLU |
| Loss | Square-root inverse-frequency weighted cross-entropy |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Dropout | 0.1 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Training budget | 30 epochs |
| Saved checkpoint | Epoch 20 |
| Label rule | Tuned class log-score offsets `[0, 1, 0]` |

The class-1 offset makes prediabetes more competitive when choosing a label. It does not add new evidence to the input data or turn the scores into calibrated probabilities.

## What about the other models?

| Frozen comparison | Test macro F1 | Diabetes recall |
|---|---:|---:|
| Always predict no diabetes | 0.3048 | 0% |
| Logistic regression baseline | 0.3930 | 17.4% |
| MLP baseline | 0.4128 | 23.3% |
| Selected weighted logistic variant | 0.4556 | 61.9% |
| Selected weighted MLP | 0.4675 | 51.0% |
| Selected boosted-tree variant | 0.4699 | 49.7% |

CatBoost scored slightly higher on the test set than the chosen MLP. The MLP had won during validation, so we retained it. Choosing again after seeing test scores would make the test part of model selection.

## How much uncertainty is there?

A paired bootstrap repeatedly resampled the test outcomes while keeping the two models' predictions paired. Across 2,000 resamples, the 95% interval for macro-F1 improvement over the baseline MLP was approximately **+0.0467 to +0.0630**.

This supports improvement on this split and these fitted models. It does not cover new training seeds, the original model search, profile clustering or future populations.

**Evidence:** [test result CSV](https://github.com/NewOsakA/diabetes-ml/blob/main/results/test_results.csv) · [per-class results](https://github.com/NewOsakA/diabetes-ml/blob/main/results/champion_test_classes.csv) · [technical report](https://github.com/NewOsakA/diabetes-ml/blob/main/results/REPORT.md)

**Next:** [What happened with broader experiments? →](Experiment-2-Results)
