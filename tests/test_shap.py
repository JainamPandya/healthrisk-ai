import pytest
import pandas as pd
from healthrisk.patient_explanation import explain_patient, _clean_feature_name
from tests.test_preprocessing import sample_data
from healthrisk.predictor import HealthRiskPredictor

@pytest.fixture
def trained_predictor(sample_data):
    predictor = HealthRiskPredictor()
    sample_data.loc[1, "early_readmission"] = 1
    predictor.train(sample_data)
    return predictor

def test_clean_feature_name():
    assert _clean_feature_name("numerical__number_inpatient") == "Number of inpatient visits"
    assert _clean_feature_name("categorical__race_Caucasian") == "Race = Caucasian"
    assert _clean_feature_name("categorical__discharge_disposition_id_1") == "Discharge disposition = 1"
    assert _clean_feature_name("categorical__diag_1_486") == "Diagnosis 1 = 486"

def test_explain_patient(trained_predictor, sample_data):
    patient = sample_data.drop(columns=["readmitted", "early_readmission"]).iloc[[0]]
    result = explain_patient(trained_predictor.model, patient)
    
    assert "risk_score" in result
    assert "risk_category" in result
    assert "top_features" in result
    assert "factors_increasing" in result
    assert "factors_decreasing" in result
    
    # Assert dataframes are returned for factors
    assert isinstance(result["top_features"], pd.DataFrame)
    assert isinstance(result["factors_increasing"], pd.DataFrame)
    assert isinstance(result["factors_decreasing"], pd.DataFrame)
