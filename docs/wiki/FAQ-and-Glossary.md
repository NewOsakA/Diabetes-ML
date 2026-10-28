# Common questions

## Is the improved model worse because its accuracy is lower?

It depends on the goal. It identifies more actual diabetes cases but makes more errors elsewhere. That is why we show accuracy, class precision/recall, balanced accuracy and macro F1 together. There is no single score that captures every consequence.

## Why not use the model with the highest test score?

We use validation data to choose. If we inspect test scores and choose again, the test becomes part of model selection. The originally selected Experiment 1 MLP therefore remains the selected model even though the boosted-tree comparison scored slightly higher on those test rows.

## Why are 0.4648 and 0.4659 both reported for Experiment 2?

**0.4648** is the mean of the three outer-fold macro-F1 values. **0.4659** is macro F1 after pooling all out-of-fold predictions. F1 is nonlinear, so averaging scores and scoring combined predictions can differ.

## Does the 13.2% improvement mean accuracy rose by 13.2 points?

No. That number describes the **relative macro-F1 increase** from 0.4128 to 0.4675 in Experiment 1. Accuracy actually decreased from 84.8% to 80.5%.

## Does “calibrated” mean medically validated?

No. Statistical calibration concerns whether model scores match observed class frequencies on evaluation data. Clinical validation is a much broader question. Better calibration does not establish medical usefulness.

## Why is prediabetes so hard?

It is rare, the survey features are limited, and some input profiles occur with different labels. These observations suggest possible difficulties, but the study does not establish a single causal explanation for the poor performance.

## Why not merge prediabetes and diabetes into one class?

That would create a different, binary task. It may be worth a separate experiment, but its scores would not be directly comparable to the three-class results. The present study keeps prediabetes visible so its weakness is not hidden by merging labels.

## Is oversampling the same as obtaining more data?

No. Repeating rows adds no independent respondents. Synthetic sampling creates approximations from existing training examples. Neither replaces genuinely new, informative observations.

## Why are there skipped tests after cloning?

Raw Kaggle data are excluded from Git. Two checks require that file and skip if it is absent. Follow [Quick Start](Quick-Start) to download it. A fresh Experiment 2 clone with the extra dependencies runs 12 checks and skips 2; with the raw data, all 14 checks run.

## Why does the Experiment 2 runner refuse to run?

It protects the committed completed results from accidental replacement. For a new full run, use another working copy, back up its completed `experiments/v2` directory, and start with a new empty results directory. `--resume` is for interrupted runs with matching code/protocol, not permission to change a completed experiment.

## Where do I find the latest project and earlier experiments?

The default `main` branch contains both completed experiments. Experiment 1 keeps its evidence in `models/` and `results/`; Experiment 2 uses `experiments/v2/`. The commands `diabetes_ml.predict` and `diabetes_ml.v2_predict` load their respective models. Including Experiment 2 does not automatically change the default predictor.

The [`experiment-1`](https://github.com/NewOsakA/diabetes-ml/tree/experiment-1) tag bookmarks the project before integration. The [`experiment-2`](https://github.com/NewOsakA/diabetes-ml/tree/experiment-2) tag bookmarks the integrated project. Future experiments can be developed on branches and merged through pull requests when complete. Tags preserve milestones, while `main` continues to move forward.

Repository visibility and wiki publication are separate from merging code. GitHub stores wiki pages in their own Git repository.

# Glossary

| Term | Simple meaning |
|---|---|
| Feature | An input column used to make a prediction |
| Target / label | The known answer the model tries to predict |
| Class | One answer category: 0, 1 or 2 here |
| Supervised learning | Learning from examples with known labels |
| Class imbalance | Unequal numbers of examples across classes |
| Baseline | A simple or fixed approach used as a comparison |
| MLP | Multilayer perceptron, a type of neural network |
| Parameter | A learned value inside a model |
| Hyperparameter | A setting chosen around training, such as depth or learning rate |
| Learning rate | Controls the size of optimizer updates |
| Batch | A group of examples used for a training update |
| Epoch | A training pass through the examples or configured sampling budget |
| Loss | A numerical penalty used during training |
| Backpropagation | Computes how network parameters influence loss |
| AdamW | The optimizer used for neural updates |
| Regularization | Techniques intended to reduce overfitting |
| Dropout | Randomly disables some hidden activations during training |
| Weight decay | Penalizes large parameter values |
| Checkpoint | Saved model state at a particular training stage |
| Boosting | Builds a collection of models in sequence using an objective |
| Soft voting | Averages class scores from multiple models |
| Argmax | Chooses the class with the largest score |
| Decision offset | An adjustment used when choosing labels from scores |
| Calibration | Adjusts scores to better match observed class frequencies |
| Overfitting | Learning patterns that fit practice data better than new data |
| Data leakage | Information enters fitting/selection that should be reserved for evaluation |
| Stratification | Keeps class proportions approximately similar across splits |
| Fold | One subset used during cross-validation |
| Nested cross-validation | Inner folds choose settings; outer folds evaluate that process |
| Out-of-fold prediction | Prediction made while that row's outer fold is excluded from fitting and selection |
| Seed | A starting value that controls randomized operations |
| Standard deviation | Describes spread across values; not automatically a confidence interval |
| Bootstrap | Repeated resampling used to study uncertainty under stated assumptions |
| Profile | A combination of recorded feature values; not necessarily a unique person |
| Inference | Using an already-trained model to make predictions |
| Artifact | A saved result, model, figure, configuration or other output |
| Repository | A project and its version history |
| Branch | A separate line of changes within a repository |

For the score definitions, return to [Understanding Metrics](Understanding-Metrics).
