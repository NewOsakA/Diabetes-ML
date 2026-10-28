# Diabetes ML

A machine learning study of diabetes status using Alex Teboul's BRFSS 2015 Diabetes Health Indicators dataset. The project investigates class imbalance through weighted neural networks, focal loss, boosted trees, multiclass decision tuning and soft voting, with model selection based on macro F1.

**Start with [the experiment report](results/REPORT.md)** or open the [executed notebook](notebooks/01_exploration_and_results.ipynb).

The validation-selected weighted MLP achieved **test macro F1 0.4675**, compared with **0.4128** for the local baseline MLP. Diabetes recall improved from **23.3% to 51.0%**; accuracy decreased from **84.8% to 80.5%**. Prediabetes recall was only **6.3%**, with **4.4%** precision. These tradeoffs are documented rather than hidden by overall accuracy.

The `experiment/robust-prediabetes` branch adds a [development-only v2 study](experiments/v2/REPORT.md): nested cross-validation, a broader MLP/CatBoost search, three neural training seeds, class-balanced sampling, random oversampling, SMOTENC, independent probability calibration, and profile/subgroup analyses. Historical test scores and original model artifacts are preserved.

The v2 study uses three outer folds and two inner folds, with separate fit, stopping, calibration and decision portions inside every training partition. A recall-focused procedure tests an experimental 5% prediabetes precision floor; its achieved outer-fold precision is reported rather than assumed. No new external test set is claimed. Install its dependencies with `.venv/bin/python -m pip install -e '.[dev,experiments]'`.

```bash
# Run once after downloading the raw dataset; allow the full nested search to finish.
.venv/bin/python -u -m diabetes_ml.v2_run
.venv/bin/python -m diabetes_ml.v2_report
# Use the saved development model after the experiment completes.
.venv/bin/python -m diabetes_ml.v2_predict \
  --input experiments/v2/example_inputs.csv \
  --output experiments/v2/example_predictions.csv \
  --policy macro
```

The v2 report builder requires the local, Git-ignored `experiments/v2/cache/` evidence created by the experiment. The generated report, executed analysis notebook, metrics, figures and final fitted models are committed for inspection without rerunning training. Completed experiments reject accidental overwrite; `--resume` can reuse inner-fold caches only when the saved protocol and source hashes match.

## Contents

```text
Diabetes ML/
├── README.md
├── METHODOLOGY.md                   # Frozen search protocol and references
├── DATA_DICTIONARY.md               # All 21 predictors and target coding
├── pyproject.toml
├── requirements.lock.txt            # Exact tested environment
├── data/
│   ├── raw/                        # Original Kaggle archive, 3-class CSV, checksums
│   └── processed/                  # Saved split IDs and inference example
├── src/diabetes_ml/
│   ├── data.py                     # Download, validation, fixed data splits
│   ├── common.py                   # Metrics, decisions, paired bootstrap
│   ├── neural.py                   # MLP, weighted CE and focal loss
│   ├── train.py                    # Complete experiment and test freeze
│   ├── predict.py                  # Batch inference with saved champion
│   ├── plots.py                    # Scientific figures
│   └── build_report.py             # Rebuild report/notebook from saved evidence
├── notebooks/01_exploration_and_results.ipynb
├── models/                         # Saved scaler, all models, registry, manifest
├── results/
│   ├── REPORT.md
│   ├── experiment_results.csv      # All 36 tuning/selection decision candidates
│   ├── test_results.csv            # Pre-frozen test comparisons
│   ├── champion_test_predictions.csv
│   ├── training.log
│   ├── data_audit.json
│   ├── environment.json
│   ├── histories/
│   └── figures/
└── tests/                          # Metrics, loss, partitions and saved inference
```

## Use the project

Clone the repository and set up the Python environment using the instructions below:

```bash
git clone --branch experiment/robust-prediabetes https://github.com/NewOsakA/diabetes-ml.git
cd diabetes-ml
```

Trained models, saved split IDs, the executed notebook and measured results are included. The Python environment, raw Kaggle files and intermediate score caches are excluded from Git. After installing the environment, run these commands from the project directory:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m diabetes_ml.predict \
  --input data/processed/example_inputs.csv \
  --output results/example_predictions.csv
```

Input CSVs must contain all 21 named feature columns from `DATA_DICTIONARY.md`. Extra columns are ignored. Use BRFSS survey categories: **Age must be 1–13, not age in years**. Inference rejects missing, nonfinite, fractional and out-of-range inputs. Outputs contain a predicted class, a class label and three raw model scores. Scores are not calibrated clinical probabilities, and decisions incorporate saved class offsets.

## Set up the environment

Tested with Python 3.12.14 on Apple Silicon macOS, CPU only. The exact package versions are recorded in `requirements.lock.txt` and `results/environment.json`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

For other platforms, install from `pyproject.toml` with `.venv/bin/python -m pip install -e '.[dev]'`; resolve any platform-specific package wheels and retain the resulting version record. Small numerical differences may occur across platforms or library versions.

Saved-model inference works immediately after installation. To run the two data-dependent integration tests or execute the analysis notebook, first download the dataset:

```bash
.venv/bin/python -c 'from diabetes_ml.data import download; download()'
.venv/bin/python -m pytest -q
```

The original experiment has four core tests and two data-dependent integration tests. Without the raw dataset, those integration tests are skipped; downloading it enables all six original checks. The v2 branch adds tests for partition isolation, calibration, sampling, selection and model artifacts. Downloading the data does not retrain models or change saved test results.

## Run the experiment

The script automatically downloads the public Kaggle archive if the source CSV is absent, verifies its schema and checksum, and runs the full experiment. No Kaggle credentials were needed for this public archive during this run. A changed endpoint or authentication policy may require placing the three-class CSV in `data/raw/` and preserving its provenance in `source.json`.

The existing results are protected against accidental reruns. To run the experiment again and overwrite the saved results using the same candidate family and seed:

```bash
.venv/bin/python -u -m diabetes_ml.train --seed 42 --overwrite > results/training.log 2>&1
.venv/bin/python -m diabetes_ml.build_report
.venv/bin/python -m pytest -q
```

The completed training run took about two minutes on this machine; timing depends on hardware. The experiment uses four CPU threads. A fresh output directory does not require `--overwrite`.

To open/re-execute the notebook using this environment, register its kernel:

```bash
.venv/bin/python -m ipykernel install --prefix .venv --name diabetes-ml --display-name 'Diabetes ML (.venv)'
```

Select that kernel in your notebook editor. The saved notebook already includes outputs and does not retrain models.

## Evaluation protocol

The stratified seed-42 partition is 80% training, 5% validation tuning, 5% validation selection and 10% final test. Standardization and class weights use training data only. The tuning portion selects checkpoints, iterations, ensemble membership and class decision offsets. The selection portion chooses the final model on macro F1. A timestamped manifest freezes model and offsets before test evaluation.

The boosted-tree finalist scored slightly higher on the exposed test set than the validation-selected champion. We retain the preselected model; selecting a replacement using those test scores would bias the final comparison. Bootstrap intervals and a profile-overlap sensitivity check are included in the report.

All model comparisons use identical data partitions. For future technique changes, use a new evaluation protocol or nested cross-validation instead of tuning repeatedly against this test set.

## Data and interpretation

Dataset: [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). File: `diabetes_012_health_indicators_BRFSS2015.csv`, containing 253,680 rows, 21 inputs and target `Diabetes_012` (0 no diabetes/pregnancy only, 1 prediabetes, 2 diabetes). Only this three-class file is extracted; the downloaded archive also contains binary variants that are not used.

This is an educational research experiment predicting self-reported status in a 2015 survey. It does not predict future onset or provide a diagnosis. No survey weighting, prospective validation, fairness audit or clinical calibration was performed. See the report for the measured limitations and the methodology for technique references.
