"""
Counterfactual explanations for HealthRisk AI.

Generates hypothetical input changes that could reduce a patient's
predicted readmission risk. Uses a greedy perturbation search over the
existing preprocessing pipeline and trained LightGBM model.

These are hypothetical scenarios for informational purposes only.
They do not constitute medical advice.
"""

import numpy as np
import pandas as pd


# Features that should never be perturbed because they are
# not meaningfully actionable.
IMMUTABLE_FEATURES = frozenset({
    "age",
    "gender",
    "race",
    "weight",
})

_DISCLAIMER = (
    "These are hypothetical scenarios for informational purposes "
    "only. They do not constitute medical advice."
)


def _get_feature_metadata(model):
    """
    Extract numerical/categorical feature lists and valid
    categorical values from the fitted preprocessing pipeline.
    """

    preprocessor = model.named_steps["preprocessor"]

    numerical_features = []
    categorical_features = []
    categorical_valid_values = {}

    for name, transformer, columns in preprocessor.transformers_:

        if name == "numerical":
            numerical_features = list(columns)

        elif name == "categorical":
            categorical_features = list(columns)

            # The encoder is the second step in the categorical
            # pipeline (imputer → encoder).
            encoder = transformer.named_steps["encoder"]

            for feature, categories in zip(
                columns,
                encoder.categories_,
            ):
                categorical_valid_values[feature] = [
                    str(c) for c in categories
                ]

    return (
        numerical_features,
        categorical_features,
        categorical_valid_values,
    )


def _generate_numerical_candidates(original_value):
    """
    Generate candidate perturbation values for a numerical feature.

    For count-like features (integers ≥ 0), candidates include zero
    and several fractions/multiples of the original value, all
    clipped to non-negative integers.
    """

    candidates = set()

    # Always try zero (e.g., 0 emergency visits).
    candidates.add(0)

    if original_value != 0:

        for multiplier in [0.25, 0.50, 0.75, 1.25, 1.50, 2.0]:

            candidate = original_value * multiplier

            # Keep integer type for integer originals.
            if isinstance(original_value, (int, np.integer)):
                candidate = int(round(candidate))

            candidate = max(0, candidate)
            candidates.add(candidate)

    # Small additive perturbations for low-value features.
    for delta in [-2, -1, 1, 2]:

        candidate = original_value + delta

        if isinstance(original_value, (int, np.integer)):
            candidate = int(candidate)

        candidate = max(0, candidate)
        candidates.add(candidate)

    # Remove the original value itself.
    candidates.discard(original_value)

    return list(candidates)


def generate_counterfactual(
    model,
    patient_df,
    target_risk=0.20,
    max_changes=5,
):
    """
    Generate a counterfactual explanation for one patient.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline (preprocessor + classifier).
    patient_df : pd.DataFrame
        A single-row DataFrame with the patient's raw features,
        in the same format the API receives.
    target_risk : float, optional
        The target readmission probability to reach. Default 0.20
        (the boundary between Low and Moderate risk).
    max_changes : int, optional
        Maximum number of features to change. Default 5.

    Returns
    -------
    dict
        original_risk, counterfactual_risk, target_achieved,
        changes (list of dicts), and disclaimer.
    """

    # ── Original risk ──────────────────────────────────────────
    original_risk = float(
        model.predict_proba(patient_df)[0, 1]
    )

    if original_risk <= target_risk:
        return {
            "original_risk": original_risk,
            "counterfactual_risk": original_risk,
            "target_achieved": True,
            "changes": [],
            "disclaimer": _DISCLAIMER,
        }

    # ── Feature metadata from the fitted pipeline ──────────────
    (
        numerical_features,
        categorical_features,
        categorical_valid_values,
    ) = _get_feature_metadata(model)

    patient_columns = set(patient_df.columns)

    # Only perturb features that are present AND mutable.
    mutable_numerical = [
        f for f in numerical_features
        if f in patient_columns
        and f not in IMMUTABLE_FEATURES
    ]

    mutable_categorical = [
        f for f in categorical_features
        if f in patient_columns
        and f not in IMMUTABLE_FEATURES
    ]

    # ── Greedy search ──────────────────────────────────────────
    current_df = patient_df.copy()
    current_risk = original_risk
    changes = []
    changed_features = set()

    for _ in range(max_changes):

        best_candidate_df = None
        best_candidate_risk = current_risk
        best_candidate_change = None

        # --- Numerical candidates ---
        for feature in mutable_numerical:
            if feature in changed_features:
                continue

            original_value = current_df[feature].iloc[0]
            candidates = _generate_numerical_candidates(
                original_value
            )

            for candidate_value in candidates:

                trial_df = current_df.copy()
                trial_df[feature] = candidate_value

                trial_risk = float(
                    model.predict_proba(trial_df)[0, 1]
                )

                if trial_risk < best_candidate_risk:
                    best_candidate_risk = trial_risk
                    best_candidate_df = trial_df
                    best_candidate_change = {
                        "feature": feature,
                        "original_value": _to_native(
                            original_value
                        ),
                        "counterfactual_value": _to_native(
                            candidate_value
                        ),
                    }

        # --- Categorical candidates ---
        for feature in mutable_categorical:
            if feature in changed_features:
                continue

            original_value = str(
                current_df[feature].iloc[0]
            )
            valid_values = categorical_valid_values.get(
                feature, []
            )

            for candidate_value in valid_values:
                if candidate_value == original_value:
                    continue

                trial_df = current_df.copy()
                trial_df[feature] = candidate_value

                trial_risk = float(
                    model.predict_proba(trial_df)[0, 1]
                )

                if trial_risk < best_candidate_risk:
                    best_candidate_risk = trial_risk
                    best_candidate_df = trial_df
                    best_candidate_change = {
                        "feature": feature,
                        "original_value": _to_native(
                            original_value
                        ),
                        "counterfactual_value": _to_native(
                            candidate_value
                        ),
                    }

        # No improvement found — stop.
        if best_candidate_change is None or best_candidate_df is None:
            break

        # Accept the best single-feature change.
        current_df = best_candidate_df
        current_risk = best_candidate_risk
        changes.append(best_candidate_change)
        changed_features.add(
            best_candidate_change["feature"]
        )

        # Target reached — stop early.
        if current_risk <= target_risk:
            break

    return {
        "original_risk": original_risk,
        "counterfactual_risk": current_risk,
        "target_achieved": current_risk <= target_risk,
        "changes": changes,
        "disclaimer": _DISCLAIMER,
    }


def _to_native(value):
    """
    Convert numpy types to native Python types for JSON
    serialisation.
    """

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value
