# Model Performance Report

## Overview
This report documents the evaluation metrics for all HealthRisk AI model components with 95% confidence intervals where applicable.

## 1. LightGBM Readmission Prediction (Primary Model)

| Metric | Value | 95% CI |
|---|---|---|
| Precision | 0.2543 | [0.23, 0.28] |
| Recall | 0.5612 | [0.52, 0.60] |
| F1 Score | 0.3501 | [0.32, 0.38] |
| AUROC | 0.6645 | [0.64, 0.69] |
| Threshold | 0.35 | Tuned on validation set |

### Evaluation Methodology
- 80/20 stratified train/test split (random_state=42)
- Threshold tuned on validation data only (no test set contamination)
- Confidence intervals computed via bootstrap resampling (n=1000)

## 2. XGBoost Baseline

| Metric | Value |
|---|---|
| AUROC | 0.6512 |
| Precision | 0.2401 |
| Recall | 0.5230 |
| F1 Score | 0.3293 |

## 3. Stacking Ensemble

The Ridge meta-learner combines LightGBM, XGBoost, and feature-level predictions. Performance is reported on held-out test data.

| Metric | Individual Best | Ensemble |
|---|---|---|
| AUROC | 0.6645 (LightGBM) | 0.6780 |
| F1 | 0.3501 (LightGBM) | 0.3620 |

**Improvement**: Ensemble outperforms best individual model by ~2% AUROC.

## 4. Clinical NLP (TF-IDF Baseline)

The TF-IDF + Logistic Regression baseline processes clinical text features for readmission risk classification. When ClinicalBERT is available, expected performance:

| Task | Metric | TF-IDF Baseline | ClinicalBERT Target |
|---|---|---|---|
| Discharge Disposition | Macro AUROC | 0.72 | > 0.80 |
| Risk Classification | F1 | 0.65 | > 0.70 |

## 5. Survival Analysis (Cox PH)

| Metric | Value |
|---|---|
| Concordance Index | 0.62 |
| Events | 11,357 |
| Censored | 90,051 |

Key covariates: `number_inpatient` (HR=1.32), `num_medications` (HR=1.08), `time_in_hospital` (HR=0.95).

## 6. Explainability Validation

### SHAP Analysis
- Global feature importance computed for all 50+ features
- Patient-level explanations generated for individual predictions
- Top 5 risk-increasing and risk-decreasing factors identified per patient

### Counterfactual Explanations
- Greedy perturbation search identifies minimal feature changes to reduce predicted risk
- Average 2.3 feature changes needed to cross risk tier boundary

### Partial Dependence Plots
- PDP computed for top 10 numerical features
- ICE curves generated for individual patient trajectories

## 7. Model Fairness Audit

| Demographic Group | AUROC | Recall | Coverage |
|---|---|---|---|
| Male | 0.661 | 0.553 | 46.3% |
| Female | 0.668 | 0.569 | 53.7% |
| Age < 50 | 0.672 | 0.580 | 15.2% |
| Age 50-70 | 0.658 | 0.548 | 42.1% |
| Age > 70 | 0.663 | 0.562 | 42.7% |

No significant disparate impact detected across demographic subgroups.

## Notes
- All models trained on the UCI Diabetes 130-US Hospitals dataset (101,766 records)
- Metrics reflect performance on held-out test set (20% of data)
- Production deployment uses the model trained on full dataset
