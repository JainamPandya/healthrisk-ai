from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from healthrisk.predictor import HealthRiskPredictor



app = FastAPI(
    title="HealthRisk AI",
    description="Early hospital readmission risk prediction API",
    version="1.0.0",
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request


MODEL_FILE = Path(
    "models/healthrisk_lightgbm.joblib"
)


predictor = HealthRiskPredictor()


# Load the trained model when available; otherwise initialise a baseline predictor
if MODEL_FILE.exists():
    predictor.load(MODEL_FILE)
else:
    # Minimal training data to initialize predictor in CI/testing environments
    _init_df = pd.DataFrame({
        "race": ["Caucasian", "AfricanAmerican", "Caucasian", "Other"],
        "gender": ["Male", "Female", "Male", "Female"],
        "age": ["[50-60)", "[60-70)", "[70-80)", "[40-50)"],
        "weight": ["?", "?", "?", "?"],
        "admission_type_id": [1, 2, 1, 3],
        "discharge_disposition_id": [1, 3, 1, 1],
        "admission_source_id": [7, 7, 7, 1],
        "time_in_hospital": [2, 10, 5, 3],
        "payer_code": ["MC", "MC", "SP", "BC"],
        "medical_specialty": ["InternalMedicine", "InternalMedicine", "Cardiology", "Family/GeneralPractice"],
        "num_lab_procedures": [20, 60, 40, 30],
        "num_procedures": [1, 4, 2, 0],
        "num_medications": [5, 18, 12, 8],
        "number_outpatient": [0, 2, 0, 1],
        "number_emergency": [0, 3, 1, 0],
        "number_inpatient": [0, 4, 1, 0],
        "diag_1": ["486", "250.83", "414.01", "250"],
        "diag_2": ["401.9", "250.01", "276", "401"],
        "diag_3": ["250", "255", "428", "272"],
        "number_diagnoses": [3, 9, 5, 4],
        "max_glu_serum": ["None", ">300", "None", "Norm"],
        "A1Cresult": ["None", ">8", "None", "Norm"],
        "metformin": ["No", "Steady", "No", "No"],
        "repaglinide": ["No", "No", "No", "No"],
        "nateglinide": ["No", "No", "No", "No"],
        "chlorpropamide": ["No", "No", "No", "No"],
        "glimepiride": ["No", "No", "No", "No"],
        "acetohexamide": ["No", "No", "No", "No"],
        "glipizide": ["No", "No", "No", "No"],
        "glyburide": ["No", "No", "No", "No"],
        "tolbutamide": ["No", "No", "No", "No"],
        "pioglitazone": ["No", "No", "No", "No"],
        "rosiglitazone": ["No", "No", "No", "No"],
        "acarbose": ["No", "No", "No", "No"],
        "miglitol": ["No", "No", "No", "No"],
        "troglitazone": ["No", "No", "No", "No"],
        "tolazamide": ["No", "No", "No", "No"],
        "examide": ["No", "No", "No", "No"],
        "citoglipton": ["No", "No", "No", "No"],
        "insulin": ["No", "Up", "No", "Steady"],
        "glyburide-metformin": ["No", "No", "No", "No"],
        "glipizide-metformin": ["No", "No", "No", "No"],
        "glimepiride-pioglitazone": ["No", "No", "No", "No"],
        "metformin-rosiglitazone": ["No", "No", "No", "No"],
        "metformin-pioglitazone": ["No", "No", "No", "No"],
        "change": ["No", "Ch", "No", "No"],
        "diabetesMed": ["No", "Yes", "No", "Yes"],
        "readmitted": ["NO", "<30", ">30", "NO"],
        "early_readmission": [0, 1, 0, 0],
    })
    predictor.train(_init_df)


class PatientData(BaseModel):
    race: str
    gender: str
    age: str
    weight: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesMed: str


@app.get("/")
def root():
    return {
        "message": "HealthRisk AI API is running",
        "model": "LightGBM",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
    }


@app.post("/predict")
def predict(patient: PatientData):

    try:

        data = patient.model_dump()

        # Convert API field names back to dataset names
        data["glyburide-metformin"] = data.pop(
            "glyburide_metformin"
        )

        data["glipizide-metformin"] = data.pop(
            "glipizide_metformin"
        )

        data["glimepiride-pioglitazone"] = data.pop(
            "glimepiride_pioglitazone"
        )

        data["metformin-rosiglitazone"] = data.pop(
            "metformin_rosiglitazone"
        )

        data["metformin-pioglitazone"] = data.pop(
            "metformin_pioglitazone"
        )

        patient_df = pd.DataFrame(
            [data]
        )

        result = predictor.predict(
            patient_df
        )

        explanation = predictor.explain(
            patient_df
        )

        increasing = []

        for _, row in explanation["factors_increasing"].iterrows():

            increasing.append({
                "feature": row["readable_feature"],
                "shap_value": float(row["shap_value"]),
            })


        decreasing = []

        for _, row in explanation["factors_decreasing"].iterrows():

            decreasing.append({
                "feature": row["readable_feature"],
                "shap_value": float(row["shap_value"]),
            })


        return {
            "risk_score": result["risk_score"],
            "risk_category": result["risk_category"],
            "factors_increasing": increasing,
            "factors_decreasing": decreasing,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(
    directory="templates"
)

@app.get("/app", response_class=HTMLResponse)
def application(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )