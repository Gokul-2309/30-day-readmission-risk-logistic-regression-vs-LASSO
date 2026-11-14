# Diabetes Patient Readmission Risk: Logistic Regression, LASSO and a Bayesian Hierarchical Model

Two analysis projects on the same public dataset. Both predict whether a hospital patient is readmitted within 30 days of discharge.

**Data:** [UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). The data file is not included in this repository. Download `diabetic_data.csv` and place it in a `data/` folder.

## Project 1: Logistic regression vs LASSO

**Question:** Which patient and admission factors are linked to 30-day readmission, and does LASSO give a simpler model that predicts as well as standard logistic regression?

- **Cleaning:** removed discharges to death or hospice, kept only each patient's first encounter, removed unknown gender.
- **Models:** logistic regression (statsmodels) for odds ratios with confidence intervals, and LASSO logistic regression (scikit-learn) with the penalty chosen by cross-validation.
- **Evaluation:** held-out test set, AUC, Brier score, ROC curve, calibration plot.
- **Code:** `readmission_glm_lasso.py`
- **Results:** `specialty_baselines.csv, bayes_odds_ratios.csv, posterior_summary.csv, model_performance.csv, trace.png, specialty_baselines.png, bayes_odds_ratios.png, sigma_posterior.png

## Project 2: Bayesian hierarchical model

**Question:** Do medical specialties differ in baseline readmission risk after adjusting for patient factors, and how uncertain are those differences?

- **Model:** Bayesian logistic regression with a varying intercept for each medical specialty (partial pooling), so specialties with little data are pulled toward the overall average.
- **Sampler:** Hamiltonian Monte Carlo implemented in NumPy, with R-hat and effective sample size diagnostics.
- **Code:** `bayesian_hmc.py` (uses the cleaning function from `readmission_glm_lasso.py`, so keep both files in the same folder).
- **Results:** in the `outputs_bayes/` folder (specialty baselines, odds ratios, posterior summary, diagnostics, plots).

## How to run

```bash
pip install -r requirements.txt
python readmission_glm_lasso.py --data data/diabetic_data.csv --out outputs
python bayesian_hmc.py --data data/diabetic_data.csv --out outputs_bayes
```

`bayesian_hmc.py` uses a random sample of patients to keep the run time short. Change it with `--n` (`--n 0` uses all patients).

## Limitations

- The data are from 1999-2008 and cover patients with diabetes only, so results may not apply to other patients or to current practice.
- The results show associations, not causes.
- Predicting readmission is hard, so predictive accuracy is modest.
