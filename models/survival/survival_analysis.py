"""
Survival Analysis Module for HealthRisk AI.

Implements time-to-event prediction models for healthcare-financial
applications:
- Cox Proportional Hazards (baseline)
- Kaplan-Meier estimator
- Deep survival model wrapper (DeepSurv/DeepHit when pycox available)

Prediction tasks:
- Time-to-readmission for hospital patients
- Time-to-complication for chronic disease patients
- Time-to-financial-covenant-breach for hospital credit risk

References:
- Cox (1972) Regression Models and Life-Tables
- Katzman et al. (2018) DeepSurv
- Lee et al. (2018) DeepHit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------------

def kaplan_meier_estimate(
    durations: np.ndarray,
    event_observed: np.ndarray,
    label: str = "Overall",
) -> Dict[str, object]:
    """
    Compute Kaplan-Meier survival curve.

    Parameters
    ----------
    durations : np.ndarray
        Time durations until event or censoring.
    event_observed : np.ndarray
        Binary indicator (1 = event occurred, 0 = censored).
    label : str
        Label for the survival curve.

    Returns
    -------
    dict
        median_survival, survival_at_timepoints, survival_table.
    """
    if LIFELINES_AVAILABLE:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=event_observed, label=label)

        survival_table = kmf.survival_function_.reset_index()
        survival_table.columns = ["time", "survival_probability"]

        return {
            "label": label,
            "median_survival": float(kmf.median_survival_time_)
                if not np.isinf(kmf.median_survival_time_) else None,
            "survival_at_30": float(kmf.predict(30))
                if 30 <= durations.max() else None,
            "survival_at_90": float(kmf.predict(90))
                if 90 <= durations.max() else None,
            "survival_at_365": float(kmf.predict(365))
                if 365 <= durations.max() else None,
            "n_events": int(event_observed.sum()),
            "n_censored": int(len(event_observed) - event_observed.sum()),
            "survival_table": survival_table.to_dict("records"),
        }

    # Manual Kaplan-Meier when lifelines is not available
    return _manual_kaplan_meier(durations, event_observed, label)


def _manual_kaplan_meier(
    durations: np.ndarray,
    event_observed: np.ndarray,
    label: str,
) -> Dict[str, object]:
    """Manual Kaplan-Meier calculation without lifelines."""
    # Sort by duration
    order = np.argsort(durations)
    t = durations[order]
    e = event_observed[order]

    unique_times = np.unique(t[e == 1])
    survival_probs = []
    current_prob = 1.0
    n_at_risk = len(t)

    for time_point in unique_times:
        events_at_t = np.sum((t == time_point) & (e == 1))
        censored_before_t = np.sum((t < time_point) & (e == 0))
        n_at_risk -= censored_before_t
        if n_at_risk > 0:
            current_prob *= (1 - events_at_t / n_at_risk)
        n_at_risk -= events_at_t
        survival_probs.append({"time": float(time_point), "survival_probability": round(current_prob, 4)})

    return {
        "label": label,
        "median_survival": None,
        "n_events": int(event_observed.sum()),
        "n_censored": int(len(event_observed) - event_observed.sum()),
        "survival_table": survival_probs,
    }


# ---------------------------------------------------------------------------
# Cox Proportional Hazards Model
# ---------------------------------------------------------------------------

class CoxPHModel:
    """
    Cox Proportional Hazards model for time-to-event prediction.

    h(t|X) = h₀(t) × exp(β'X)

    Uses lifelines.CoxPHFitter when available.
    """

    def __init__(self, penalizer: float = 0.01):
        """
        Parameters
        ----------
        penalizer : float
            L2 regularisation strength (higher = more regularisation).
        """
        self.penalizer = penalizer
        self.model = None
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> Dict[str, object]:
        """
        Fit the Cox PH model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with covariates, duration, and event columns.
        duration_col : str
            Column name for time-to-event.
        event_col : str
            Column name for event indicator (1=event, 0=censored).

        Returns
        -------
        dict
            Model summary with concordance index and coefficients.
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required for CoxPHModel. "
                "Install with: pip install lifelines"
            )

        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(df, duration_col=duration_col, event_col=event_col)
        self.is_fitted = True

        # Extract results
        summary_df = self.model.summary
        coefficients = {}
        for idx, row in summary_df.iterrows():
            coefficients[str(idx)] = {
                "coef": round(float(row["coef"]), 4),
                "exp_coef": round(float(row["exp(coef)"]), 4),
                "p_value": round(float(row["p"]), 6),
            }

        return {
            "concordance_index": round(float(self.model.concordance_index_), 4),
            "n_observations": int(self.model.event_observed.shape[0]),
            "n_events": int(self.model.event_observed.sum()),
            "coefficients": coefficients,
        }

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict relative hazard for new observations.

        Parameters
        ----------
        df : pd.DataFrame
            New observations (same columns as training data, excluding
            duration and event).

        Returns
        -------
        np.ndarray
            Partial hazard values exp(β'X).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_partial_hazard(df).values.flatten()

    def predict_survival(
        self,
        df: pd.DataFrame,
        times: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Predict survival function for new observations.

        Parameters
        ----------
        df : pd.DataFrame
            New observations.
        times : list of float, optional
            Time points at which to evaluate survival.

        Returns
        -------
        pd.DataFrame
            Survival probabilities at each time point.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_survival_function(df, times=times)


# ---------------------------------------------------------------------------
# Readmission Survival Analysis
# ---------------------------------------------------------------------------

def prepare_readmission_survival_data(
    df: pd.DataFrame,
    readmission_col: str = "early_readmission",
    time_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare a patient DataFrame for survival analysis of readmission.

    If no explicit time column is available, uses time_in_hospital
    as a proxy duration and readmission as the event.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data.
    readmission_col : str
        Column indicating readmission event.
    time_col : str, optional
        Column for time duration. If None, uses 'time_in_hospital'.
    feature_cols : list of str, optional
        Feature columns to include. If None, selects numeric columns.

    Returns
    -------
    pd.DataFrame
        Survival-ready DataFrame with 'duration' and 'event' columns.
    """
    result = pd.DataFrame()

    # Duration
    if time_col and time_col in df.columns:
        result["duration"] = df[time_col].astype(float)
    elif "time_in_hospital" in df.columns:
        result["duration"] = df["time_in_hospital"].astype(float)
    else:
        result["duration"] = np.ones(len(df))

    # Ensure duration > 0
    result["duration"] = result["duration"].clip(lower=0.5)

    # Event
    if readmission_col in df.columns:
        result["event"] = df[readmission_col].astype(int)
    else:
        result["event"] = np.zeros(len(df), dtype=int)

    # Features
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {readmission_col, "duration", "event", time_col or ""}
        feature_cols = [c for c in feature_cols if c not in exclude]

    for col in feature_cols:
        if col in df.columns:
            result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return result
