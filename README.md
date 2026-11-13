# Predicting 30-day hospital readmission: logistic regression vs LASSO

**Data:** UCI "Diabetes 130-US Hospitals for Years 1999-2008" (`diabetic_data.csv`), https://archive.ics.uci.edu/dataset/296

**Question:** Which patient and admission factors are associated with readmission within 30 days, and does LASSO give a simpler model that predicts as well as classical logistic regression?

## How to run

```bash
pip install -r requirements.txt
mkdir data    # put diabetic_data.csv inside data/
python readmission_glm_lasso.py --data data/diabetic_data.csv --out outputs
```

## Method

1. **Cleaning:** removed discharges to death or hospice, kept the first encounter of each patient (so rows are independent), removed unknown gender. See `outputs/cleaning_log.txt`.
2. **Target:** readmitted within 30 days (1) vs not (0).
3. **Features:** age, length of stay, lab tests, procedures, medications, prior outpatient, emergency and inpatient visits, number of diagnoses, primary diagnosis group (ICD-9), admission type, discharge type, payer, specialty, HbA1c, medication change.
4. **Split:** 80% train, 20% test, stratified. Numeric variables standardised using train statistics only.
5. **Model 1:** logistic regression (statsmodels) for odds ratios with 95% confidence intervals.
6. **Model 2:** LASSO logistic regression (scikit-learn), penalty chosen by 5-fold cross-validation on AUC.
7. **Evaluation:** test AUC, Brier score, ROC and calibration plots.

## Results

_Fill this in after you run the script on the real data. Use your own numbers._

| Model | Test AUC | Brier score | Predictors used |
| --- | --- | --- | --- |
| Logistic | | | |
| LASSO | | | |

Strongest predictors (from `outputs/glm_odds_ratios.csv`):
- ...

Predictors LASSO removed (from `outputs/lasso_nonzero_coefficients.csv`):
- ...

## Limitations

_Write 3 or 4 honest points in your own words._
- Data is from 1999-2008 and from US hospitals, so it may not match today's patients.
- Only the first encounter per patient was used, so frequent returners are under-represented.
- Readmission is rare (about 1 in 9), so AUC is expected to be modest.
- Associations are not causal.
