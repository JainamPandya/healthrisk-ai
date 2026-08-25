"""
Explainability package for HealthRisk AI.

Provides SHAP, Counterfactual Explanations, and Partial Dependence Plots (PDP).
"""

from healthrisk.patient_explanation import explain_patient_prediction
from healthrisk.counterfactual import generate_counterfactual
from healthrisk.pdp import (
    calculate_numerical_pdp,
    calculate_categorical_pdp,
    calculate_all_pdp,
)

__all__ = [
    "explain_patient_prediction",
    "generate_counterfactual",
    "calculate_numerical_pdp",
    "calculate_categorical_pdp",
    "calculate_all_pdp",
]
