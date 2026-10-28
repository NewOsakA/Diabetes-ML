# What do the scores mean?

**Start by asking what each score counts.** A high score for one question can hide a poor answer to another.

## Accuracy: how often was the final label correct?

Accuracy counts correct labels across all rows. In the Experiment 1 test set, **84.2%** of rows belong to “No Diabetes.” Always predicting that label therefore gives 84.2% accuracy while detecting no diabetes or prediabetes cases.

This is why we report more than accuracy.

## Recall and precision ask different questions

Using the Experiment 1 weighted MLP's **diabetes** predictions:

![Recall finds 1,801 of 3,534 actual diabetes cases; precision finds 1,801 correct cases among 4,282 diabetes predictions.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/precision-versus-recall.png)

| Question | Calculation | Result |
|---|---|---:|
| Of all actual diabetes cases, how many did we find? **Recall** | 1,801 ÷ 3,534 | **51.0%** |
| Of all diabetes predictions, how many were correct? **Precision** | 1,801 ÷ 4,282 | **42.1%** |

The numerator is the same. The denominator changes. That is the key distinction.

- **False negative:** an actual diabetes case predicted as one of the other classes.
- **False positive:** a non-diabetes-class response predicted as diabetes. Here that includes both no-diabetes and prediabetes responses.

Lowering a decision threshold may find more cases but can also create more false positives. We must inspect both precision and recall.

## How to read a confusion matrix

**Rows are actual labels. Columns are predicted labels.** The diagonal contains correct predictions.

| Actual ↓ / Predicted → | No diabetes | Prediabetes | Diabetes |
|---|---:|---:|---:|
| No diabetes | **18,597** | 453 | 2,321 |
| Prediabetes | 274 | **29** | 160 |
| Diabetes | 1,559 | 174 | **1,801** |

For example, `2,321` means 2,321 no-diabetes responses were classified as diabetes. The prediabetes row shows only 29 of 463 responses identified correctly.

## F1 and macro F1

**F1** combines a class's precision and recall using their harmonic mean. It rewards having both, rather than only one.

**Macro F1** calculates F1 separately for all three classes and averages them equally. The small prediabetes class receives the same importance as the large no-diabetes class.

For the weighted MLP:

```text
No-diabetes F1    0.8898
Prediabetes F1    0.0518
Diabetes F1      0.4608
                 ──────
Macro F1 = their average = 0.4675
```

**Macro F1 of 0.4675 does not mean 46.75% accuracy.** These measure different things. Moving from 0.4128 to 0.4675 is +0.0547 absolute, or about +13.2% relative; it is not +13.2 percentage points of accuracy.

## Other terms in the reports

| Metric | Plain-language meaning | Direction |
|---|---|---|
| Balanced accuracy | Average recall of the three classes | Higher |
| Specificity | For one class, the fraction of other-class rows correctly kept out of that class | Higher |
| Support | Number of actual examples of a class | A count, not a quality score |
| Average precision, or AP | Summarizes the precision–recall curve as we vary the threshold; measures ranking quality | Higher |
| Log loss | A probability-based penalty that particularly penalizes confident mistakes | Lower |
| Multiclass Brier score | Average summed squared difference between class scores and the true one-hot class vector | Lower |

Balanced accuracy here is `(87.0% + 6.3% + 51.0%) ÷ 3 ≈ 48.1%`. The high no-diabetes recall does not hide the low prediabetes recall.

For AP, class prevalence provides a useful no-skill reference. A rare-class AP of 0.035 is not “3.5% accuracy.” Lower log loss or Brier score improves probabilistic performance overall; a reliability plot helps inspect calibration more specifically.

**Next:** [How we keep evaluation separate →](How-We-Evaluate)
