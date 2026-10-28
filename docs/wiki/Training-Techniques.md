# What techniques did we test?

The central training question is: **how can the model pay attention to rare classes without making too many incorrect predictions?** We tested several approaches rather than assuming one would work best.

## 1. Ordinary cross-entropy

Cross-entropy penalizes assigning a low score to the correct label. It was the baseline neural-network loss. With imbalanced data, the many majority-class examples can dominate the total learning signal.

## 2. Class-weighted loss

We increase the penalty for errors involving less common classes. This changes how much examples matter during training; it does not duplicate them.

We tested **inverse-frequency weighting** and **square-root inverse-frequency weighting**. The square root makes the correction less extreme. The Experiment 1 selected MLP used square-root weights.

![Diagram: Training Techniques](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/class-weighting.png)

Greater weighting can increase minority recall while reducing precision or accuracy. It is not a free improvement.

## 3. Focal loss

Focal loss reduces the contribution of easy examples so that harder examples receive relatively more attention. The **gamma** setting controls that effect. It can be combined with class weights.

We tested gamma 2 in Experiment 1 and gamma 1 and 2 configurations in Experiment 2. Difficult examples can also be ambiguous or noisy, so emphasizing them does not guarantee improvement.

## 4. Regularization

- **Dropout:** temporarily switches off some hidden activations during training, encouraging the network to rely less on particular paths.
- **Weight decay:** discourages unnecessarily large parameter values.
- **Checkpoint selection:** retains a training stage that performs well on separate stopping data.

These aim to limit overfitting: fitting the training data better than the model generalizes to other rows.

## 5. Three sampling approaches in Experiment 2

| Approach | What changes? | What it does not provide |
|---|---|---|
| Class-balanced replacement sampling | Draws classes equally often in expectation during each neural training epoch | It does not create new observed respondents or guarantee exactly balanced individual batches |
| Random oversampling | Repeats existing prediabetes training examples | It does not add independent information |
| SMOTENC | Creates synthetic training examples while preserving categorical answer values | It does not guarantee that a generated profile describes a real person |

Random oversampling and SMOTENC increased prediabetes training count to **15% of the no-diabetes training count**. Evaluation rows kept their natural class distribution. Sampling variants used unweighted loss to avoid applying two imbalance corrections simultaneously.

SMOTENC treated binary and ordinal answers as categorical; BMI and unhealthy-day counts were numeric. All sampling was restricted to fitting rows. These sampling configurations did not produce a dependable improvement in the completed study. [SMOTENC documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)

## 6. Boosting and ensembles

CatBoost and histogram gradient boosting build collections of trees using a prediction objective. We varied depth, learning rate, regularization and class weighting. We also tested averages of tree and neural scores in Experiment 1. A blend is called **soft voting** because it combines scores rather than counting only final labels.

## 7. Calibration and decision tuning do different jobs

| Question | Technique |
|---|---|
| “Does a score match how often that class occurs?” | Calibration |
| “Which label should we choose from these scores?” | Decision tuning |

Experiment 2 fits a custom calibrator on separate rows, adjusting class log scores with positive scales and biases. It then uses another separate portion for decision tuning. Better probability scores need not produce better labels under the same decision rule. [Calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

The recall-focused policy tests an **experimental 5% prediabetes precision floor**. That is a research setting, not a recommended medical standard. It failed to reach the floor on all three outer evaluation folds.

**Next:** [Learn to read the metrics →](Understanding-Metrics)
