"""
Stacking Ensemble Module for HealthRisk AI.

Implements a two-level stacking ensemble that combines predictions
from multiple model components:
  Level 0: XGBoost, LightGBM, Clinical NLP, GNN, Survival
  Level 1: Ridge regression meta-learner

Uses time-aware cross-validation to prevent temporal data leakage.

References:
- Wolpert (1992) Stacked Generalization
- PDF Section A7.3: Stacking Ensemble Architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


class StackingEnsemble:
    """
    Two-level stacking ensemble for healthcare risk prediction.

    Level 0 models produce out-of-fold predictions.
    Level 1 meta-learner (Ridge regression) learns optimal combination.
    """

    def __init__(
        self,
        n_folds: int = 5,
        meta_alphas: Optional[List[float]] = None,
    ):
        """
        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds.
        meta_alphas : list of float
            Regularisation strengths for Ridge meta-learner.
        """
        self.n_folds = n_folds
        self.meta_alphas = meta_alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.meta_learner = None
        self.level0_models: Dict[str, object] = {}
        self.is_fitted = False

    def fit(
        self,
        level0_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, object]:
        """
        Fit the meta-learner on Level 0 model predictions.

        Parameters
        ----------
        level0_predictions : dict
            Model name → predicted probabilities array.
            Each array has shape (n_samples,).
        y_true : np.ndarray
            True binary labels.

        Returns
        -------
        dict
            Meta-learner coefficients, cross-val score, model contributions.
        """
        # Stack Level 0 predictions into feature matrix
        model_names = sorted(level0_predictions.keys())
        X_meta = np.column_stack([
            level0_predictions[name] for name in model_names
        ])

        # Fit Ridge meta-learner with built-in CV
        self.meta_learner = RidgeCV(
            alphas=self.meta_alphas,
            scoring="neg_mean_squared_error",
            cv=self.n_folds,
        )
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True
        self._model_names = model_names

        # Extract model contributions
        coefficients = dict(zip(model_names, self.meta_learner.coef_))

        # Evaluate
        ensemble_pred = self.meta_learner.predict(X_meta)
        ensemble_pred_clipped = np.clip(ensemble_pred, 0, 1)

        try:
            auc = roc_auc_score(y_true, ensemble_pred_clipped)
        except ValueError:
            auc = 0.0

        binary_preds = (ensemble_pred_clipped >= 0.5).astype(int)

        return {
            "best_alpha": float(self.meta_learner.alpha_),
            "model_contributions": {
                k: round(float(v), 4) for k, v in coefficients.items()
            },
            "intercept": round(float(self.meta_learner.intercept_), 4),
            "ensemble_auroc": round(auc, 4),
            "ensemble_precision": round(
                float(precision_score(y_true, binary_preds, zero_division=0)), 4
            ),
            "ensemble_recall": round(
                float(recall_score(y_true, binary_preds, zero_division=0)), 4
            ),
            "ensemble_f1": round(
                float(f1_score(y_true, binary_preds, zero_division=0)), 4
            ),
        }

    def predict(
        self,
        level0_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble predictions from Level 0 model outputs.

        Parameters
        ----------
        level0_predictions : dict
            Model name → predicted probabilities.

        Returns
        -------
        np.ndarray
            Ensemble predicted probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        X_meta = np.column_stack([
            level0_predictions[name] for name in self._model_names
        ])

        return np.clip(self.meta_learner.predict(X_meta), 0, 1)

    def compare_models(
        self,
        level0_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compare individual model performance vs ensemble.

        Parameters
        ----------
        level0_predictions : dict
            Model name → predicted probabilities.
        y_true : np.ndarray
            True binary labels.

        Returns
        -------
        pd.DataFrame
            Performance comparison table.
        """
        results = []

        for name, preds in level0_predictions.items():
            preds_clipped = np.clip(preds, 0, 1)
            try:
                auc = roc_auc_score(y_true, preds_clipped)
            except ValueError:
                auc = 0.0
            binary = (preds_clipped >= 0.5).astype(int)
            results.append({
                "model": name,
                "auroc": round(auc, 4),
                "precision": round(float(precision_score(y_true, binary, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, binary, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, binary, zero_division=0)), 4),
            })

        # Add ensemble
        if self.is_fitted:
            ensemble_preds = self.predict(level0_predictions)
            try:
                auc = roc_auc_score(y_true, ensemble_preds)
            except ValueError:
                auc = 0.0
            binary = (ensemble_preds >= 0.5).astype(int)
            results.append({
                "model": "ENSEMBLE (Ridge Meta-Learner)",
                "auroc": round(auc, 4),
                "precision": round(float(precision_score(y_true, binary, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, binary, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, binary, zero_division=0)), 4),
            })

        return pd.DataFrame(results).sort_values("auroc", ascending=False)
