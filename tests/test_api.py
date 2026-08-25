import pytest
from fastapi.testclient import TestClient
from healthrisk.api import app, predictor

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "HealthRisk AI API is running",
        "model": "LightGBM",
    }

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }

def test_predict_endpoint_valid_input():
    # Provide a valid input for prediction
    patient_data = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[30-40)",
        "weight": "?",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 2,
        "payer_code": "BC",
        "medical_specialty": "Family/GeneralPractice",
        "num_lab_procedures": 20,
        "num_procedures": 1,
        "num_medications": 5,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "486",
        "diag_2": "401.9",
        "diag_3": "250",
        "number_diagnoses": 3,
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
        "insulin": "No",
        "glyburide_metformin": "No",
        "glipizide_metformin": "No",
        "glimepiride_pioglitazone": "No",
        "metformin_rosiglitazone": "No",
        "metformin_pioglitazone": "No",
        "change": "No",
        "diabetesMed": "No"
    }

    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_category" in data
    assert "factors_increasing" in data
    assert "factors_decreasing" in data

def test_predict_endpoint_missing_input():
    # Sending missing input
    response = client.post("/predict", json={"age": "50"})
    assert response.status_code == 422  # Unprocessable Entity (validation error)
