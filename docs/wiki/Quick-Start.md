# Run the project, step by step

**You can read the reports and executed notebooks without installing anything.** To run predictions on your computer, follow the steps below. A “terminal” is an application where you type commands; a “repository” is the project folder plus its version history.

## 1. Get a copy

Have Python 3.12 and Git available first. Open a terminal in the folder where you want to keep the project.

```bash
git clone https://github.com/NewOsakA/diabetes-ml.git
cd diabetes-ml
```

The default `main` branch contains both completed experiments, with separate reports and saved models. No branch switch is needed.

## 2. Create an isolated Python environment

A virtual environment keeps this project's packages separate from your other Python projects. The folder `.venv` will hold that environment.

**macOS or Linux:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,experiments]'
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,experiments]'
```

The completed experiment used Python 3.12.14 on Apple Silicon macOS. `requirements.lock.txt` records the tested packages. Installing from the project configuration is more portable across operating systems, but may resolve different package versions. Exact scores can vary with versions and hardware.

The commands below use macOS/Linux paths. On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

## 3. Try the saved Experiment 1 model

```bash
.venv/bin/python -m diabetes_ml.predict --input data/processed/example_inputs.csv --output example_predictions.csv
```

The input already contains correctly named feature columns. The output is a small CSV you can open in a spreadsheet or text editor. This uses the selected weighted MLP from Experiment 1 and does not train anything.

## 4. Try the saved Experiment 2 model

```bash
.venv/bin/python -m diabetes_ml.v2_predict --input experiments/v2/example_inputs.csv --output v2_predictions.csv --policy macro
```

This loads the final development-trained CatBoost artifact. Its predictions demonstrate model loading; they are not a fresh performance test.

For the separate recall-focused research policy, use `--policy recall`. That policy did not meet its precision target on the outer folds, so its name should not be interpreted as evidence that it performs better.

## 5. Understand the output

| Column | Meaning |
|---|---|
| `predicted_class` | `0` no diabetes/pregnancy only; `1` prediabetes; `2` diabetes |
| `predicted_label` | The readable name of that class |
| Columns containing `score_class_` | One score for each class; their exact prefix differs between the two predictors |

Class offsets can affect the chosen label without changing the reported underlying score vector. A model score is not a validated individual medical probability.

For your own input CSV, include all 21 columns from the [Dataset Guide](Dataset-Guide). **Use survey codes, especially `Age` categories 1–13.** The code rejects missing, fractional or out-of-range inputs. Extra columns, including the target, are ignored as predictors.

## 6. Download data and run the checks

The large raw Kaggle file is not stored in Git. Saved-model prediction does not require it. Data-dependent checks and data exploration do.

```bash
.venv/bin/python -c 'from diabetes_ml.data import download; download()'
.venv/bin/python -m pytest -q
```

The completed local study passed **14 tests**. A fresh clone with Experiment 2 dependencies but no raw dataset runs **12 tests and skips 2** data-dependent checks. Downloading the data enables those two checks.

## 7. Open the notebooks

The notebooks already contain outputs and are readable on GitHub. To execute them in a notebook editor, select the project environment or register its kernel:

```bash
.venv/bin/python -m ipykernel install --prefix .venv --name diabetes-ml --display-name 'Diabetes ML (.venv)'
```

The Experiment 2 analysis notebook reads committed metrics, figures and saved models. Its **report-building script** additionally requires the local OOF cache created by the full experiment; that cache is excluded from Git.

## Training again is optional

The completed results are protected from accidental replacement. Experiment 1 accepts an explicit `--overwrite`; Experiment 2 rejects a completed output directory.

For a fresh full Experiment 2 run, use a separate working copy, move its existing `experiments/v2` results folder to a distinct backup name, and create a new empty `experiments/v2` folder. Then run:

```bash
.venv/bin/python -u -m diabetes_ml.v2_run
.venv/bin/python -m diabetes_ml.v2_report
```

The recorded run took about 11 minutes on the recorded machine; yours may differ. An interrupted run can reuse matching inner-fold caches with `--resume`. Changed experimental code or settings require a separate protocol/output history.

**Next:** [Find your way around the files →](Code-and-Files)
