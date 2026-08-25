"""
Tests for counterfactual explanations.
"""

from pathlib import Path

import pandas as pd
import pytest

from healthrisk.predictor import HealthRiskPredictor
from healthrisk.counterfactual import (
    generate_counterfactual,
    IMMUTABLE_FEATURES,
    _get_feature_metadata,
    _generate_numerical_candidates,
)


MODEL_FILE = Path("models/healthrisk_lightgbm.joblib")


def _make_high_risk_patient():
    """
    Construct a sample patient likely to trigger a high risk score.
    """

    return pd.DataFrame([{
        "race": "AfricanAmerican",
        "gender": "Male",
        "age": "[70-80)",
        "weight": "?",
        "admission_type_id": 1,
        "discharge_disposition_id": 3,
        "admission_source_id": 7,
        "time_in_hospital": 10,
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 60,
        "num_procedures": 4,
        "num_medications": 18,
        "number_outpatient": 2,
        "number_emergency": 3,
        "number_inpatient": 4,
        "diag_1": "250.83",
        "diag_2": "250.01",
        "diag_3": "255",
        "number_diagnoses": 9,
        "max_glu_serum": ">300",
        "A1Cresult": ">8",
        "metformin": "Steady",
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
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
    }])


def _make_low_risk_patient():
    """
    Construct a sample patient likely to trigger a low risk score.
    """

    return pd.DataFrame([{
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
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "No",
        "diabetesMed": "No",
    }])


@pytest.fixture(scope="module")
def predictor():
    """Load the trained model once for all tests."""

    if not MODEL_FILE.exists():
        pytest.skip("Trained model not found")

    p = HealthRiskPredictor()
    p.load(MODEL_FILE)
    return p


# ── Structure tests ────────────────────────────────────────────


class TestOutputStructure:

    def test_result_has_required_keys(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        assert "original_risk" in result
        assert "counterfactual_risk" in result
        assert "target_achieved" in result
        assert "changes" in result
        assert "disclaimer" in result

    def test_disclaimer_present(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        assert "not constitute medical advice" in result[
            "disclaimer"
        ]

    def test_changes_have_required_fields(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        for change in result["changes"]:
            assert "feature" in change
            assert "original_value" in change
            assert "counterfactual_value" in change


# ── Behavior tests ─────────────────────────────────────────────


class TestCounterfactualBehavior:

    def test_risk_is_reduced(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        assert result["counterfactual_risk"] <= result[
            "original_risk"
        ]

    def test_changed_features_exist_in_input(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        patient_columns = set(patient.columns)

        for change in result["changes"]:
            assert change["feature"] in patient_columns

    def test_immutable_features_not_changed(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(patient)

        for change in result["changes"]:
            assert change["feature"] not in IMMUTABLE_FEATURES

    def test_max_changes_respected(self, predictor):
        patient = _make_high_risk_patient()

        result = predictor.counterfactual(
            patient,
            max_changes=2,
        )

        assert len(result["changes"]) <= 2

    def test_categorical_values_are_valid(self, predictor):
        """Changed categorical features use values from
        the training data."""

        patient = _make_high_risk_patient()

        _, cat_features, valid_values = _get_feature_metadata(
            predictor.model
        )

        result = predictor.counterfactual(patient)

        for change in result["changes"]:
            if change["feature"] in cat_features:
                assert str(
                    change["counterfactual_value"]
                ) in valid_values[change["feature"]]


# ── Edge case tests ────────────────────────────────────────────


class TestEdgeCases:

    def test_already_low_risk_returns_no_changes(
        self, predictor
    ):
        patient = _make_low_risk_patient()

        original_risk = float(
            predictor.model.predict_proba(patient)[0, 1]
        )

        # Use a very high target so the patient is already
        # below it.
        result = predictor.counterfactual(
            patient,
            target_risk=0.99,
        )

        assert result["target_achieved"] is True
        assert result["changes"] == []

    def test_model_not_loaded_raises(self):
        empty_predictor = HealthRiskPredictor()
        patient = _make_high_risk_patient()

        with pytest.raises(RuntimeError, match="not been trained"):
            empty_predictor.counterfactual(patient)


# ── Numerical candidate generation ────────────────────────────


class TestNumericalCandidates:

    def test_zero_is_a_candidate(self):
        candidates = _generate_numerical_candidates(5)
        assert 0 in candidates

    def test_original_value_excluded(self):
        candidates = _generate_numerical_candidates(4)
        assert 4 not in candidates

    def test_no_negative_candidates(self):
        candidates = _generate_numerical_candidates(1)
        assert all(c >= 0 for c in candidates)


# ── SHAP regression test ──────────────────────────────────────


class TestNoRegression:

    def test_shap_explain_still_works(self, predictor):
        """Existing SHAP explanation must still return the
        same structure after adding counterfactual support."""

        patient = _make_high_risk_patient()

        result = predictor.explain(patient)

        assert "risk_score" in result
        assert "risk_category" in result
        assert "top_features" in result
        assert "factors_increasing" in result
        assert "factors_decreasing" in result

    def test_predict_still_works(self, predictor):
        """Existing predict must still return the same
        structure after adding counterfactual support."""

        patient = _make_high_risk_patient()

        result = predictor.predict(patient)

        assert "risk_score" in result
        assert "risk_category" in result
        assert isinstance(result["risk_score"], float)
