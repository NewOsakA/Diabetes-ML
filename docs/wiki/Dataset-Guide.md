# What is in the dataset?

Think of the CSV as a large spreadsheet. **Each row is one survey response. Each input column describes an answer or health indicator.** The target column contains the status we want the model to predict.

We use `diabetes_012_health_indicators_BRFSS2015.csv` from [Alex Teboul's Kaggle dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). It has **253,680 rows**, **21 input features**, and one target column.

## The answer the model learns to predict

| Target code | Meaning in this dataset | Rows | Approximate share |
|---|---|---:|---:|
| `0` | No diabetes, including diabetes only during pregnancy | 213,703 | 84.24% |
| `1` | Prediabetes | 4,631 | 1.83% |
| `2` | Diabetes | 35,346 | 13.93% |

The target column is named **`Diabetes_012`**. We remove it from the input features before fitting a model. Otherwise the model would already have the answer.

![Rounded class balance in a group of 100 responses.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/class-balance.png)

This unequal group size is called **class imbalance**. “Class” simply means one of the three answer categories.

## What information does the model receive?

| Group | Inputs | What they describe |
|---|---|---|
| Reported health history | `HighBP`, `HighChol`, `Stroke`, `HeartDiseaseorAttack` | Blood pressure, cholesterol and reported cardiovascular history |
| Health checks and access | `CholCheck`, `AnyHealthcare`, `NoDocbcCost` | Cholesterol checking, coverage and cost-related barriers |
| Body size and daily health | `BMI`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk` | Body mass index, general health, unhealthy days and walking difficulty |
| Lifestyle | `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump` | Smoking history, activity, diet and alcohol indicators |
| Demographic categories | `Sex`, `Age`, `Education`, `Income` | Coded demographic answers |

See the [complete data dictionary](https://github.com/NewOsakA/diabetes-ml/blob/main/DATA_DICTIONARY.md) for each column's coding.

## Numbers can mean categories

Three important examples:

- **`Age = 9` means the 60–64 age band.** It does not mean nine years old. Valid age codes run from 1 to 13.
- **`GenHlth = 1` means excellent and `5` means poor.** The numbers are ordered ratings.
- **`Smoker = 1` means at least 100 cigarettes over a lifetime.** It does not necessarily mean currently smoking.

Zero is a valid answer in many columns. We do not replace every zero with a missing value.

## What we checked

The source file has no missing cells. The code checks its column names, numeric ranges, target labels and SHA-256 checksum. A checksum is a file fingerprint that helps detect an unexpected change to the data.

The file also contains repeated rows and repeated input profiles. Two people can give the same coarse survey answers, so matching rows do not prove they are the same person. We kept the rows and investigated this in the [evaluation](How-We-Evaluate).

## What is absent?

This file does not provide blood glucose, HbA1c, person identifiers or survey-design weights. We cannot add those measurements by guessing them, and these processed sample proportions are not survey-weighted national estimates.

The task describes **reported status at the time of a survey**. It does not measure the chance that a person will develop diabetes later.

**Next:** [How models learn →](How-Models-Learn)
