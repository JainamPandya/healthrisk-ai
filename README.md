# HealthRisk AI

## Overview
HealthRisk AI is a machine-learning intelligence platform designed to predict early hospital readmission risks for patients. By analyzing patient demographics, medical history, lab procedures, and medications, the system estimates the probability of a patient being readmitted to the hospital within 30 days of discharge. 

**⚠️ IMPORTANT MEDICAL DISCLAIMER:** This project is a machine-learning risk estimation tool, intended strictly for research and analytical demonstration purposes. It is **NOT** a medical diagnosis, and its outputs should **NOT** be used for clinical decision-making or to replace professional medical judgment.

## Key Features
* **LightGBM Prediction Engine:** Fast, gradient-boosting framework optimized for binary classification.
* **XGBoost Comparison:** Includes baseline XGBoost modeling for performance benchmarking.
* **Threshold Tuning:** Custom decision boundary tuning prioritizing recall over raw accuracy.
* **SHAP Explainability (Global & Patient-level):** Uses Shapley Additive Explanations to demystify black-box predictions, presenting human-readable feature importance for individual patients.
* **Counterfactual Explanations:** Automatically calculates hypothetical adjustments (e.g., changes in lab procedures or medications) that would lower a patient's predicted risk score.
* **Partial Dependence Plots (PDP):** Visualizes the marginal effect of numerical features (like the number of inpatient visits) on the predicted risk globally.
* **FastAPI API:** Robust RESTful backend serving predictions and explainability data.
* **Web Interface:** Includes a frontend interface (`/app`) for interacting with the model.
* **Automated Tests:** Comprehensive Pytest suite covering preprocessing, predictions, SHAP, and API endpoints (66% overall coverage; ~100% core module coverage).
* **Docker Support:** Fully containerized for easy, reproducible deployments.

## Tech Stack
* **Machine Learning:** Scikit-learn, LightGBM, XGBoost, SHAP
* **Data Processing:** Pandas, NumPy
* **Backend:** FastAPI, Uvicorn, Pydantic, Jinja2
* **Testing:** Pytest, Pytest-cov
* **DevOps:** Docker

## Dataset Description
The model is trained on a diabetic patient dataset. The dataset includes categorical features (race, gender, admission sources, medical specialty, diagnoses, and specific diabetes medications) and numerical features (age brackets mapped to numerical ranges, time in hospital, number of lab procedures, medications, and previous inpatient/outpatient visits). The target variable is `early_readmission` (binary), indicating readmission within 30 days.

## Model Evaluation & Performance
The final deployed model is a LightGBM classifier. Because hospital readmission datasets are inherently imbalanced (most patients are not readmitted early), raw accuracy is a misleading metric. Missing a high-risk patient (false negative) is typically more costly than flagging a low-risk patient (false positive). Therefore, we apply **Threshold Tuning** to prioritize **Recall**.

**LightGBM Final Metrics (Post-Tuning):**
* **Accuracy:** 0.6572
* **Precision:** 0.1851
* **Recall:** 0.6090
* **F1-Score:** 0.2839
* **ROC-AUC:** 0.6862
* **PR-AUC:** 0.2386

*Note: These metrics reflect the dataset's complexity and class imbalance. They are not clinically validated performance standards.*

## Explainability 

### 1. SHAP (Shapley Additive Explanations)
SHAP is used to explain *why* the model made a specific prediction for an individual patient. The API returns a list of factors that increased the patient's risk and factors that decreased it, converting raw model features into human-readable text (e.g., "Number of inpatient visits" instead of `number_inpatient`).

### 2. Counterfactual Explanations
Instead of just asking "why", counterfactuals ask "how to change it." The system generates a hypothetical patient profile, altering mutable features (e.g., medications or time in hospital) while keeping immutable features (like age or race) constant, to demonstrate what would theoretically reduce the patient's risk to a "Low" category.

### 3. Partial Dependence Plots (PDP)
PDPs show the global relationship between specific numerical features and the model's predicted risk. For example, a PDP plot for `number_inpatient` visualizes how increasing inpatient visits globally correlates with a higher predicted probability of readmission.

## Local Setup & Installation

**1. Create and activate a virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```powershell
pip install -e .
```

**3. Run the API locally:**
```powershell
uvicorn healthrisk.api:app --reload
```

**4. Run the automated test suite:**
```powershell
python -m pytest tests/ -v
```

## Docker Usage

**1. Build the Docker image:**
```bash
docker build -t healthrisk-ai .
```

**2. Run the Docker container:**
```bash
docker run -p 8000:8000 healthrisk-ai
```
*(If you need environment variables, append `--env-file .env`)*

## API Endpoints

Once the application (local or Docker) is running, you can access:
* **Root**: `GET /` - Basic API info.
* **Health Check**: `GET /health` - Verifies the service is running and the LightGBM model is loaded in memory.
* **Web UI**: `GET /app` - Interactive frontend.
* **Prediction**: `POST /predict` - Accepts a JSON payload and returns the risk score, risk category, and SHAP factors.

### Example `/predict` Payload
```json
{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "weight": "?",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 16,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "8",
    "diag_2": "250.43",
    "diag_3": "403",
    "number_diagnoses": 7,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "Up",
    "glyburide_metformin": "No",
    "glipizide_metformin": "No",
    "glimepiride_pioglitazone": "No",
    "metformin_rosiglitazone": "No",
    "metformin_pioglitazone": "No",
    "change": "Ch",
    "diabetesMed": "Yes"
}
```

## Architecture
See [Architecture Documentation](docs/architecture.md) for a detailed system diagram and data flow.

## Model Details
See the [Model Card](docs/model_card.md) for detailed information on model limitations, ethical considerations, and out-of-scope uses.
