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


# Load the trained model when the API starts
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

predictor.load(
    MODEL_FILE
)


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