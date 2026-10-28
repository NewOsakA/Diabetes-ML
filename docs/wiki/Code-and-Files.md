# Where is everything?

You do not need to read every CSV or Python module. Start with the document that answers your question.

| I want to… | Open… |
|---|---|
| Understand the project | [Wiki home](Home) |
| Understand each input field | [Data dictionary](https://github.com/NewOsakA/diabetes-ml/blob/main/DATA_DICTIONARY.md) |
| Read Experiment 1's full findings | [Experiment 1 report](https://github.com/NewOsakA/diabetes-ml/blob/main/results/REPORT.md) |
| Read Experiment 2's full findings | [Experiment 2 report](https://github.com/NewOsakA/diabetes-ml/blob/experiment/robust-prediabetes/experiments/v2/REPORT.md) |
| See every Experiment 1 comparison | [Test results CSV](https://github.com/NewOsakA/diabetes-ml/blob/main/results/test_results.csv) |
| See individual Experiment 2 evaluation folds | [Outer results CSV](https://github.com/NewOsakA/diabetes-ml/blob/experiment/robust-prediabetes/experiments/v2/outer_results.csv) |
| Check Experiment 2's exact settings | [Protocol JSON](https://github.com/NewOsakA/diabetes-ml/blob/experiment/robust-prediabetes/experiments/v2/protocol.json) |

## A beginner's folder map

```text
data/                 Inputs and saved row partitions
src/diabetes_ml/      Python code
models/               Experiment 1 trained models and scaler
results/              Experiment 1 metrics, figures and report
notebooks/            Experiment 1 exploration and results notebook
experiments/v2/       Experiment 2 results, figures, models and notebook
tests/                Automated checks
docs/wiki/            Editable source copy of this wiki and its images
```

A **CSV** is a table. **JSON** stores named settings or records. **NPZ** stores arrays. **IPYNB** is a notebook containing code and outputs. **PT**, **CBM** and **joblib** files hold saved model artifacts, depending on the library.

## Follow the code in this order

| Module | Responsibility |
|---|---|
| `data.py` | Download and check the data; create the Experiment 1 split |
| `common.py` | Class names, metrics, decision rules and bootstrap calculations |
| `neural.py` | Experiment 1 MLP architecture, losses and training |
| `train.py` | Run Experiment 1 and freeze comparisons before test evaluation |
| `predict.py` | Use the saved Experiment 1 model |
| `v2_protocol.py` | Define candidates, data roles and development-only boundaries |
| `v2_models.py` | Fit, sample, calibrate, save and load Experiment 2 models |
| `v2_run.py` | Coordinate the nested study and diagnostics |
| `v2_predict.py` | Use a saved Experiment 2 policy |
| `v2_report.py` | Build the Experiment 2 report, figures and notebook from local evidence |

These files are in [src/diabetes_ml on the experiment branch](https://github.com/NewOsakA/diabetes-ml/tree/experiment/robust-prediabetes/src/diabetes_ml).

## Which numbers belong to which stage?

![Diagram: Code and Files](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/evidence-map.png)

`inner_results.csv` contains model-selection estimates. `outer_results.csv` contains the separate evaluation results for the nested study. `oof_summary.csv` pools outer evaluation predictions. A final saved model's selection score is not a new test score.

`cache/` contains large intermediate arrays and row-ID manifests used locally. It is excluded from Git. The report, figures, final models and summarized evidence are committed.

**Next:** [Common questions and definitions →](FAQ-and-Glossary)
