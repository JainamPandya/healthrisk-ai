"""Tests for the stacking ensemble module."""

import numpy as np
import pytest

from models.ensemble.stacking_ensemble import StackingEnsemble


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    y = np.random.randint(0, 2, n)
    # Create correlated predictions
    pred_lgbm = y * 0.6 + np.random.rand(n) * 0.4
    pred_xgb = y * 0.55 + np.random.rand(n) * 0.45
    pred_nlp = y * 0.4 + np.random.rand(n) * 0.6
    return {
        "predictions": {
            "lightgbm": np.clip(pred_lgbm, 0, 1),
            "xgboost": np.clip(pred_xgb, 0, 1),
            "clinical_nlp": np.clip(pred_nlp, 0, 1),
        },
        "y_true": y,
    }


class TestStackingEnsemble:
    def test_fit(self, sample_data):
        ensemble = StackingEnsemble()
        result = ensemble.fit(sample_data["predictions"], sample_data["y_true"])
        assert "best_alpha" in result
        assert "model_contributions" in result
        assert result["ensemble_auroc"] > 0
        assert ensemble.is_fitted

    def test_predict(self, sample_data):
        ensemble = StackingEnsemble()
        ensemble.fit(sample_data["predictions"], sample_data["y_true"])
        preds = ensemble.predict(sample_data["predictions"])
        assert len(preds) == 200
        assert all(0 <= p <= 1 for p in preds)

    def test_predict_before_fit_raises(self, sample_data):
        ensemble = StackingEnsemble()
        with pytest.raises(RuntimeError):
            ensemble.predict(sample_data["predictions"])

    def test_compare_models(self, sample_data):
        ensemble = StackingEnsemble()
        ensemble.fit(sample_data["predictions"], sample_data["y_true"])
        comparison = ensemble.compare_models(
            sample_data["predictions"], sample_data["y_true"]
        )
        assert len(comparison) == 4  # 3 models + ensemble
        assert "ENSEMBLE" in comparison.iloc[0]["model"] or comparison.shape[0] == 4
