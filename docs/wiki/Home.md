# Diabetes ML, explained simply

**Can health and lifestyle survey answers help a model distinguish no diabetes, prediabetes and diabetes?** This project compares several machine learning approaches and studies how they behave when one group is much smaller than the others.

You do not need to know Python or machine learning to read this wiki. Start with the picture and follow the reading path below.

![About 84 of 100 responses have no diabetes, 2 have prediabetes and 14 have diabetes. These counts are rounded.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/class-balance.png)

> **The central challenge:** most responses belong to “No Diabetes.” A model can look accurate while missing almost everyone in the smaller classes.

## What the project found

| Question | Answer in plain language |
|---|---|
| Did weighting the neural network help? | In Experiment 1, diabetes recall rose from **23.3% to 51.0%**, while accuracy fell from **84.8% to 80.5%**. |
| Did the broader Experiment 2 search help? | It gave a **small** average macro-F1 gain: **0.4621 → 0.4648**, using the same outer folds for comparison. |
| Is prediabetes classification reliable? | **No.** Precision and recall remain low, and the recall-focused rule missed its precision target in all three outer folds. |
| Can these predictions diagnose someone? | No. This is a research project classifying reported survey status, not a validated medical tool or a forecast of future illness. |

## Pick a reading path

**Understand the findings:** [Dataset Guide](Dataset-Guide) → [Understanding Metrics](Understanding-Metrics) → [Experiment 1 Results](Experiment-1-Results) → [Experiment 2 Results](Experiment-2-Results).

**Learn how it works:** [How Models Learn](How-Models-Learn) → [Training Techniques](Training-Techniques) → [How We Evaluate](How-We-Evaluate).

**Run it yourself:** [Quick Start](Quick-Start) → [Code and Files](Code-and-Files) → [FAQ and Glossary](FAQ-and-Glossary).

**Understand the boundaries:** [Limitations and Next Steps](Limitations-and-Next-Steps).

## Two experiments, two different questions

![Diagram: Home](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/two-experiments.png)

**Do not directly rank an Experiment 1 test score against an Experiment 2 cross-validation score.** Their evaluation data and procedures differ.

The repository's [main branch](https://github.com/NewOsakA/diabetes-ml) contains both completed experiments. Each keeps its own reports, results and saved models. The default predictor still uses Experiment 1's selected MLP; Experiment 2 has its own prediction command. See [Quick Start](Quick-Start) to try either one.

For a permanent snapshot, use the [`experiment-1`](https://github.com/NewOsakA/diabetes-ml/tree/experiment-1) tag for the project before integration or [`experiment-2`](https://github.com/NewOsakA/diabetes-ml/tree/experiment-2) for the integrated project. A tag is a bookmark to a particular version.

## Evidence, not just a summary

- [Experiment 1 technical report](https://github.com/NewOsakA/diabetes-ml/blob/main/results/REPORT.md)
- [Experiment 2 technical report](https://github.com/NewOsakA/diabetes-ml/blob/main/experiments/v2/REPORT.md)
- [Executed Experiment 1 notebook](https://github.com/NewOsakA/diabetes-ml/blob/main/notebooks/01_exploration_and_results.ipynb)
- [Executed Experiment 2 notebook](https://github.com/NewOsakA/diabetes-ml/blob/main/experiments/v2/analysis.ipynb)
- [Dataset by Alex Teboul on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

**Next:** [Understand the dataset →](Dataset-Guide)
