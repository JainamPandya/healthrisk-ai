import pytest
import pandas as pd
from pathlib import Path
from healthrisk.predictor import HealthRiskPredictor
from tests.test_preprocessing import sample_data  # reuse fixture

@pytest.fixture
def trained_predictor(sample_data):
    # Train a very simple model using sample data
    predictor = HealthRiskPredictor()
    # Need to make sure there are both classes in target to train a binary classifier
    # So we manually set early_readmission to have both 0 and 1
    sample_data.loc[1, "early_readmission"] = 1
    predictor.train(sample_data)
    return predictor

def test_predictor_train(trained_predictor):
    assert trained_predictor.model is not None
    assert hasattr(trained_predictor.model, "predict_proba")

def test_predictor_predict_uninitialized():
    predictor = HealthRiskPredictor()
    with pytest.raises(RuntimeError, match="Model has not been trained"):
        predictor.predict(pd.DataFrame())

def test_predictor_predict(trained_predictor, sample_data):
    # Predict on the first patient
    patient = sample_data.drop(columns=["readmitted", "early_readmission"]).iloc[[0]]
    result = trained_predictor.predict(patient)
    
    assert "risk_score" in result
    assert "risk_category" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_category"] in ["Low", "Moderate", "High"]

def test_predictor_explain_uninitialized():
    predictor = HealthRiskPredictor()
    with pytest.raises(RuntimeError, match="Model has not been trained"):
        predictor.explain(pd.DataFrame())

def test_predictor_counterfactual_uninitialized():
    predictor = HealthRiskPredictor()
    with pytest.raises(RuntimeError, match="Model has not been trained"):
        predictor.counterfactual(pd.DataFrame())

def test_predictor_save_load(trained_predictor, tmp_path):
    filepath = tmp_path / "model.joblib"
    trained_predictor.save(filepath)
    assert filepath.exists()
    
    new_predictor = HealthRiskPredictor()
    new_predictor.load(filepath)
    assert new_predictor.model is not None

def test_predictor_save_uninitialized(tmp_path):
    predictor = HealthRiskPredictor()
    with pytest.raises(RuntimeError, match="Model has not been trained"):
        predictor.save(tmp_path / "model.joblib")
