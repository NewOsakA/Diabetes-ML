# Data dictionary

Source: [Alex Teboul's Kaggle dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), cleaned BRFSS 2015 survey responses. The raw CSV is preserved in `data/raw/`. No patient identifier or survey-design weights are available in this file.

| Column | Meaning / coding |
|---|---|
| Diabetes_012 | Target: 0 = no diabetes or diabetes only during pregnancy; 1 = prediabetes; 2 = diabetes |
| HighBP | Reported high blood pressure: 0 no, 1 yes |
| HighChol | Reported high cholesterol: 0 no, 1 yes |
| CholCheck | Cholesterol checked in the last five years: 0 no, 1 yes |
| BMI | Body mass index, represented as integer values in the provided file |
| Smoker | Smoked at least 100 cigarettes over one's lifetime: 0 no, 1 yes; **not current smoking status** |
| Stroke | Reported stroke history: 0 no, 1 yes |
| HeartDiseaseorAttack | Reported coronary heart disease or myocardial infarction: 0 no, 1 yes |
| PhysActivity | Physical activity in past 30 days, excluding job: 0 no, 1 yes |
| Fruits | Fruit at least once per day: 0 no, 1 yes |
| Veggies | Vegetables at least once per day: 0 no, 1 yes |
| HvyAlcoholConsump | Heavy alcohol consumption indicator: 0 no, 1 yes |
| AnyHealthcare | Any healthcare coverage: 0 no, 1 yes |
| NoDocbcCost | Needed a doctor in past 12 months but could not see one because of cost: 0 no, 1 yes |
| GenHlth | General health rating: 1 excellent, 2 very good, 3 good, 4 fair, 5 poor |
| MentHlth | Days mental health was not good in the past 30 days: 0–30 |
| PhysHlth | Days physical health was not good in the past 30 days: 0–30 |
| DiffWalk | Serious difficulty walking or climbing stairs: 0 no, 1 yes |
| Sex | Dataset coding: 0 female, 1 male |
| Age | Age **category**, 1–13: 18–24; 25–29; 30–34; 35–39; 40–44; 45–49; 50–54; 55–59; 60–64; 65–69; 70–74; 75–79; 80+ |
| Education | Ordered education category, 1–6, from no school/kindergarten to college graduate |
| Income | Ordered income category, 1–8, from less than $10,000 to $75,000 or more, in 2015 categories |

All 21 inputs use numeric values in the source file. Linear models and MLPs standardize all inputs using training-set statistics; ordered categories are treated as ordinal scores. Tree models receive the original numeric codes. This is a consistent preprocessing choice; ordinal steps do not necessarily represent equal real-world distances.

Zero is a valid answer for many columns and must not be replaced with missing values. The observed 253,680-row file has no missing cells. Schema, integral codes, ranges, source checksum and target labels are asserted in `src/diabetes_ml/data.py`.

The task predicts **reported status in a cross-sectional survey**, not future diabetes onset or a laboratory diagnosis. Inputs do not include HbA1c or blood glucose. Results apply to this processed sample, not survey-weighted national prevalence estimates.
