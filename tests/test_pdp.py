"""
Tests for Partial Dependence Plot (PDP) functionality.
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from healthrisk.predictor import HealthRiskPredictor


MODEL_FILE = Path("models/healthrisk_lightgbm.joblib")


# ── Import tests ──────────────────────────────────────────────


class TestImports:

    def test_pdp_module_imports(self):
        """The pdp module should import without errors."""

        import healthrisk.pdp  # noqa: F401

    def test_public_functions_importable(self):
        """Key public functions should be importable."""

        from healthrisk.pdp import (  # noqa: F401
            load_model,
            get_model_features,
            validate_features,
            compute_pdp,
            plot_single_pdp,
            generate_pdp_plots,
        )

    def test_default_features_defined(self):
        from healthrisk.pdp import DEFAULT_PDP_FEATURES

        assert isinstance(DEFAULT_PDP_FEATURES, list)
        assert len(DEFAULT_PDP_FEATURES) > 0

    def test_feature_display_names_defined(self):
        from healthrisk.pdp import FEATURE_DISPLAY_NAMES

        assert isinstance(FEATURE_DISPLAY_NAMES, dict)
        assert "number_inpatient" in FEATURE_DISPLAY_NAMES

    def test_display_name_lookup(self):
        from healthrisk.pdp import _get_display_name

        assert _get_display_name("number_inpatient") == (
            "Number of Inpatient Visits"
        )

    def test_display_name_fallback(self):
        from healthrisk.pdp import _get_display_name

        # Unknown features get auto-formatted.
        result = _get_display_name("some_new_feature")
        assert result == "Some New Feature"


# ── Model loading tests ──────────────────────────────────────


class TestModelLoading:

    def test_model_file_exists(self):
        """The trained model file must exist on disk."""

        assert MODEL_FILE.exists(), (
            f"Model file not found: {MODEL_FILE}"
        )

    def test_load_model_returns_pipeline(self):
        from healthrisk.pdp import load_model

        model = load_model(MODEL_FILE)
        assert hasattr(model, "named_steps")
        assert "preprocessor" in model.named_steps
        assert "classifier" in model.named_steps

    def test_load_model_missing_file_raises(self):
        from healthrisk.pdp import load_model

        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.joblib")


# ── Feature validation tests ─────────────────────────────────


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""

    if not MODEL_FILE.exists():
        pytest.skip("Trained model not found")

    from healthrisk.pdp import load_model

    return load_model(MODEL_FILE)


class TestFeatureValidation:

    def test_requested_features_exist_in_model(self, model):
        """All default PDP features should exist in the model."""

        from healthrisk.pdp import (
            DEFAULT_PDP_FEATURES,
            get_model_features,
        )

        model_features = get_model_features(model)

        for feature in DEFAULT_PDP_FEATURES:
            assert feature in model_features, (
                f"Feature '{feature}' not found in model. "
                f"Available: {model_features}"
            )

    def test_validate_features_returns_valid(self, model):
        from healthrisk.pdp import validate_features

        valid = validate_features(
            model,
            ["number_inpatient", "time_in_hospital"],
        )

        assert "number_inpatient" in valid
        assert "time_in_hospital" in valid

    def test_validate_features_skips_invalid(self, model):
        from healthrisk.pdp import validate_features

        valid = validate_features(
            model,
            ["number_inpatient", "totally_fake_feature"],
        )

        assert "number_inpatient" in valid
        assert "totally_fake_feature" not in valid

    def test_validate_all_invalid_raises(self, model):
        from healthrisk.pdp import validate_features

        with pytest.raises(ValueError, match="None of the"):
            validate_features(
                model,
                ["fake_feature_1", "fake_feature_2"],
            )


# ── PDP computation tests ────────────────────────────────────


def _make_sample_data(n=50):
    """
    Create a minimal sample DataFrame matching the model's
    expected input schema.
    """

    rng = np.random.RandomState(42)

    return pd.DataFrame({
        "race": rng.choice(
            ["Caucasian", "AfricanAmerican"], n
        ),
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.choice(
            ["[50-60)", "[60-70)", "[70-80)"], n
        ),
        "weight": "?",
        "admission_type_id": rng.choice([1, 2, 3], n),
        "discharge_disposition_id": rng.choice(
            [1, 3, 6], n
        ),
        "admission_source_id": rng.choice([1, 7], n),
        "time_in_hospital": rng.randint(1, 14, n),
        "payer_code": rng.choice(["MC", "BC", "SP"], n),
        "medical_specialty": rng.choice(
            ["InternalMedicine", "Cardiology"], n
        ),
        "num_lab_procedures": rng.randint(10, 80, n),
        "num_procedures": rng.randint(0, 6, n),
        "num_medications": rng.randint(1, 30, n),
        "number_outpatient": rng.randint(0, 5, n),
        "number_emergency": rng.randint(0, 5, n),
        "number_inpatient": rng.randint(0, 6, n),
        "diag_1": rng.choice(
            ["250.83", "486", "414.01"], n
        ),
        "diag_2": rng.choice(
            ["250.01", "401.9", "276"], n
        ),
        "diag_3": rng.choice(["255", "250", "428"], n),
        "number_diagnoses": rng.randint(1, 16, n),
        "max_glu_serum": rng.choice(
            ["None", ">200", ">300"], n
        ),
        "A1Cresult": rng.choice(
            ["None", ">7", ">8"], n
        ),
        "metformin": rng.choice(
            ["No", "Steady", "Up"], n
        ),
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
        "insulin": rng.choice(
            ["No", "Steady", "Up", "Down"], n
        ),
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": rng.choice(["No", "Ch"], n),
        "diabetesMed": rng.choice(["Yes", "No"], n),
    })


class TestPDPComputation:

    def test_compute_pdp_returns_dict(self, model):
        from healthrisk.pdp import compute_pdp

        X = _make_sample_data(30)

        result = compute_pdp(
            model,
            X,
            "number_inpatient",
            grid_resolution=10,
        )

        assert isinstance(result, dict)
        assert "values" in result
        assert "avg_predictions" in result

    def test_pdp_values_are_probabilities(self, model):
        from healthrisk.pdp import compute_pdp

        X = _make_sample_data(30)

        result = compute_pdp(
            model,
            X,
            "number_inpatient",
            grid_resolution=10,
        )

        preds = result["avg_predictions"]

        assert np.all(preds >= 0.0), (
            "Predicted probabilities should be >= 0"
        )
        assert np.all(preds <= 1.0), (
            "Predicted probabilities should be <= 1"
        )

    def test_pdp_grid_has_correct_length(self, model):
        from healthrisk.pdp import compute_pdp

        X = _make_sample_data(30)
        resolution = 10

        result = compute_pdp(
            model,
            X,
            "time_in_hospital",
            grid_resolution=resolution,
        )

        # Grid may be <= resolution if feature has fewer
        # unique values.
        assert len(result["values"]) <= resolution
        assert len(result["values"]) == len(
            result["avg_predictions"]
        )


# ── Plot output tests ────────────────────────────────────────


class TestPlotOutput:

    def test_plot_single_pdp_creates_file(self, model, tmp_path):
        from healthrisk.pdp import (
            compute_pdp,
            plot_single_pdp,
        )

        X = _make_sample_data(30)

        result = compute_pdp(
            model,
            X,
            "number_inpatient",
            grid_resolution=10,
        )

        output_file = tmp_path / "test_pdp.png"

        plot_single_pdp(
            grid_values=result["values"],
            avg_predictions=result["avg_predictions"],
            feature_name="number_inpatient",
            output_path=output_file,
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_pdp_plots_creates_files(
        self, model, tmp_path
    ):
        from healthrisk.pdp import generate_pdp_plots

        # Use only 2 features for speed.
        features = [
            "number_inpatient",
            "time_in_hospital",
        ]

        X = _make_sample_data(30)

        # Write sample data to a temp CSV so
        # generate_pdp_plots can load it.
        data_path = tmp_path / "test_data.csv"
        X_with_targets = X.copy()
        X_with_targets["readmitted"] = "NO"
        X_with_targets["early_readmission"] = 0
        X_with_targets.to_csv(data_path, index=False)

        output_dir = tmp_path / "pdp"

        results = generate_pdp_plots(
            model=model,
            data_path=data_path,
            features=features,
            output_dir=output_dir,
            sample_size=30,
            grid_resolution=10,
        )

        assert len(results) == 2

        for entry in results:
            # Metadata structure.
            assert "feature" in entry
            assert "display_name" in entry
            assert "output_path" in entry
            assert "grid_min" in entry
            assert "grid_max" in entry
            assert "pred_min" in entry
            assert "pred_max" in entry

            # File was created.
            assert Path(entry["output_path"]).exists()
            assert (
                Path(entry["output_path"]).stat().st_size
                > 0
            )

            # Predictions are probabilities.
            assert 0.0 <= entry["pred_min"] <= 1.0
            assert 0.0 <= entry["pred_max"] <= 1.0

    def test_generate_pdp_plots_missing_data_raises(
        self, model, tmp_path
    ):
        from healthrisk.pdp import generate_pdp_plots

        with pytest.raises(FileNotFoundError):
            generate_pdp_plots(
                model=model,
                data_path=tmp_path / "no_such_file.csv",
            )


# ── No-regression tests ──────────────────────────────────────


class TestNoRegression:

    def test_predict_still_works(self):
        """Adding PDP must not break existing prediction."""

        if not MODEL_FILE.exists():
            pytest.skip("Trained model not found")

        predictor = HealthRiskPredictor()
        predictor.load(MODEL_FILE)

        patient = _make_sample_data(1)
        result = predictor.predict(patient)

        assert "risk_score" in result
        assert "risk_category" in result
        assert isinstance(result["risk_score"], float)
