# Model Card: HealthRisk AI

## Model Name
HealthRisk AI Readmission Predictor (LightGBM)

## Intended Use
This model is designed to estimate the probability that a patient will be readmitted to the hospital within 30 days of discharge. It is intended to serve as a research and analytical demonstration tool for evaluating machine learning risk estimation pipelines. It provides risk categorization alongside patient-level explainability (SHAP, Counterfactuals) to help users understand the factors driving the risk score.

## Out-of-scope Use
- **Clinical Decision Making:** The model is not a medical device and must not be used to diagnose, treat, or manage any medical conditions.
- **Automated Triage:** The model's predictions should not be used to automatically accept or deny patient care, alter medication regimens, or assign medical resources without human intervention.
- **Production Healthcare Environments:** This model has not been validated in a clinical setting or approved by any regulatory body (e.g., FDA). 

## Prediction Target
**Target Variable:** `early_readmission` (Binary classification)
- `1`: Patient readmitted within 30 days.
- `0`: Patient not readmitted within 30 days.

## Input Features
The model consumes a combination of demographic, administrative, and clinical features:
- **Demographics:** Race, Gender, Age brackets
- **Administrative:** Admission source, Discharge disposition, Payer code
- **Clinical History:** Number of inpatient/outpatient/emergency visits
- **Current Stay:** Time in hospital, Number of lab procedures, Number of procedures, Number of medications, Diagnoses (diag_1, diag_2, diag_3), Number of diagnoses
- **Lab Results:** Max glucose serum, A1C result
- **Medications:** Changes in medication (e.g., Metformin, Insulin, etc.), whether any diabetes medication was prescribed.

## Training/Evaluation Information
The model was trained on a historical diabetic patient dataset. 
- **Algorithm:** LightGBM (Gradient Boosting Framework)
- **Baseline Comparison:** XGBoost was used as a baseline to benchmark the LightGBM performance.

## Class Imbalance
The dataset exhibits significant class imbalance, which is typical in readmission datasets (the majority of patients are not readmitted early). Traditional accuracy is a misleading metric for this problem, as false negatives (missing a high-risk patient) are much more critical than false positives.

## Threshold Selection
To address the class imbalance, custom threshold tuning was applied to the decision boundary. The threshold was optimized to prioritize **Recall** over raw accuracy, ensuring the model flags as many true high-risk patients as possible.

## Evaluation Metrics
Final LightGBM metrics on the test set (Post-Tuning):
- **Accuracy:** 0.6572
- **Precision:** 0.1851
- **Recall:** 0.6090
- **F1-Score:** 0.2839
- **ROC-AUC:** 0.6862
- **PR-AUC:** 0.2386

*(Note: These metrics reflect the inherent difficulty of the readmission prediction task and are not clinical standards.)*

## Explainability 

### SHAP
Shapley Additive Explanations (SHAP) are used to provide local, patient-level explanations. For any given prediction, the system outputs the specific features that increased or decreased the patient's risk score, providing transparency into the model's decision-making process.

### Partial Dependence Plots (PDP)
PDPs are generated to analyze global model behavior. They visualize the marginal effect of numerical features (e.g., `num_lab_procedures`, `time_in_hospital`) on the predicted risk score across the entire dataset.

### Counterfactual Explanations
The system generates counterfactual explanations to answer "what-if" scenarios. By modifying mutable features (e.g., medications, length of stay) while keeping immutable features (e.g., age, race) constant, the system provides theoretical profiles that would result in a lower risk score.

## Limitations
- **Data Bias:** The model is trained on a specific historical dataset and may not generalize well to different hospital systems, demographics, or geographic regions.
- **Feature Completeness:** The model only has access to the features provided in the dataset. It lacks access to clinical notes, unstructured data, or deeper physiological indicators that human practitioners use.
- **Causality:** SHAP and Counterfactuals highlight correlations used by the model, not necessarily true medical causality.

## Risks
- **Over-reliance:** Users might incorrectly assume the risk score is a definitive clinical prognosis.
- **Misinterpretation of Explanations:** Non-technical users might misinterpret SHAP values or Counterfactuals as prescribed medical interventions rather than mathematical model sensitivities.

## Ethical Considerations
Care must be taken when interpreting predictions across different demographic groups. If the historical data contains biases in treatment patterns or readmission rates based on race or gender, the model may perpetuate these biases. The tool should be used to audit such disparities, not enforce them.

## Medical Disclaimer
**⚠️ IMPORTANT MEDICAL DISCLAIMER:** This project is a machine-learning risk estimation tool, intended strictly for research and analytical demonstration purposes. It is **NOT** a medical diagnosis, and its outputs should **NOT** be used for clinical decision-making or to replace professional medical judgment.
