"""
Pharmaceutical Analytics Module for HealthRisk AI.

Implements risk-adjusted Net Present Value (rNPV) pipeline valuation,
clinical trial signal analysis, and pharmaceutical portfolio
optimisation for health-sector equity portfolios.

References:
- DiMasi et al. (2016) R&D Cost Estimates
- Phase Success Probability Data (BIO/QLS)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Phase Success Probabilities (Industry Benchmarks)
# ---------------------------------------------------------------------------

PHASE_SUCCESS_RATES = {
    "preclinical": 0.10,
    "phase_1": 0.63,
    "phase_2": 0.31,
    "phase_3": 0.58,
    "regulatory": 0.85,
}

# Indication-specific adjustments to base rates
INDICATION_ADJUSTMENTS = {
    "oncology": {"phase_2": 0.25, "phase_3": 0.40},
    "cardiovascular": {"phase_2": 0.35, "phase_3": 0.62},
    "cns": {"phase_2": 0.22, "phase_3": 0.50},
    "infectious_disease": {"phase_2": 0.38, "phase_3": 0.65},
    "rare_disease": {"phase_2": 0.40, "phase_3": 0.60},
    "autoimmune": {"phase_2": 0.30, "phase_3": 0.55},
    "metabolic": {"phase_2": 0.35, "phase_3": 0.58},
}


# ---------------------------------------------------------------------------
# Drug Candidate
# ---------------------------------------------------------------------------

@dataclass
class DrugCandidate:
    """Represents a drug in the development pipeline."""
    name: str
    indication: str
    current_phase: str          # preclinical, phase_1, phase_2, phase_3, regulatory, marketed
    peak_sales_estimate: float  # Annual peak sales ($M)
    years_to_market: float      # Estimated years to market from current phase
    patent_years_remaining: float
    enrollment_pct: float = 1.0  # For active trials: enrollment vs target


# ---------------------------------------------------------------------------
# rNPV Valuation
# ---------------------------------------------------------------------------

def calculate_rnpv(
    candidate: DrugCandidate,
    discount_rate: float = 0.10,
    sales_ramp_years: int = 3,
    peak_duration_years: int = 5,
    decline_years: int = 3,
) -> Dict[str, float]:
    """
    Calculate risk-adjusted Net Present Value for a drug candidate.

    rNPV = Σ [P(success at stage i) × CF_i / (1+r)^t_i]

    Parameters
    ----------
    candidate : DrugCandidate
        Drug candidate to value.
    discount_rate : float
        Discount rate (default 10%).
    sales_ramp_years : int
        Years to reach peak sales after launch.
    peak_duration_years : int
        Years at peak sales.
    decline_years : int
        Years of sales decline post-peak.

    Returns
    -------
    dict
        rnpv, cumulative_success_probability, projected_cash_flows.
    """
    # Calculate cumulative probability of reaching market
    phases_remaining = _get_remaining_phases(candidate.current_phase)
    indication = candidate.indication.lower()

    cumulative_prob = 1.0
    for phase in phases_remaining:
        base_rate = PHASE_SUCCESS_RATES.get(phase, 0.5)
        # Apply indication-specific adjustment if available
        if indication in INDICATION_ADJUSTMENTS:
            adjusted = INDICATION_ADJUSTMENTS[indication].get(phase, base_rate)
            cumulative_prob *= adjusted
        else:
            cumulative_prob *= base_rate

    # Project cash flows (simplified revenue model)
    cash_flows = []
    year = candidate.years_to_market

    # Ramp-up phase
    for y in range(1, sales_ramp_years + 1):
        revenue = candidate.peak_sales_estimate * (y / sales_ramp_years)
        cash_flows.append((year + y, revenue))

    # Peak phase
    for y in range(peak_duration_years):
        t = year + sales_ramp_years + y
        cash_flows.append((t, candidate.peak_sales_estimate))

    # Decline phase
    for y in range(1, decline_years + 1):
        t = year + sales_ramp_years + peak_duration_years + y
        revenue = candidate.peak_sales_estimate * (1 - y / (decline_years + 1))
        cash_flows.append((t, revenue))

    # Discount cash flows and apply probability
    rnpv = 0.0
    for t, cf in cash_flows:
        discounted = cf / ((1 + discount_rate) ** t)
        rnpv += cumulative_prob * discounted

    return {
        "rnpv_millions": round(rnpv, 2),
        "cumulative_success_probability": round(cumulative_prob, 4),
        "current_phase": candidate.current_phase,
        "indication": candidate.indication,
        "peak_sales_estimate": candidate.peak_sales_estimate,
        "total_projected_revenue": round(sum(cf for _, cf in cash_flows), 2),
        "years_to_market": candidate.years_to_market,
    }


def _get_remaining_phases(current_phase: str) -> List[str]:
    """Return list of phases remaining before market launch."""
    all_phases = ["preclinical", "phase_1", "phase_2", "phase_3", "regulatory"]
    if current_phase == "marketed":
        return []
    if current_phase in all_phases:
        idx = all_phases.index(current_phase)
        return all_phases[idx:]
    return all_phases


# ---------------------------------------------------------------------------
# Pipeline Valuation
# ---------------------------------------------------------------------------

def value_pipeline(
    candidates: List[DrugCandidate],
    discount_rate: float = 0.10,
) -> Dict[str, object]:
    """
    Value an entire pharmaceutical pipeline.

    Parameters
    ----------
    candidates : list of DrugCandidate
        All drug candidates in the pipeline.
    discount_rate : float
        Discount rate for NPV calculation.

    Returns
    -------
    dict
        total_rnpv, candidate_values, concentration_risk.
    """
    valuations = []
    for candidate in candidates:
        val = calculate_rnpv(candidate, discount_rate)
        val["name"] = candidate.name
        valuations.append(val)

    total_rnpv = sum(v["rnpv_millions"] for v in valuations)

    # Concentration risk: largest single candidate as % of pipeline
    if total_rnpv > 0:
        max_candidate = max(valuations, key=lambda v: v["rnpv_millions"])
        concentration = max_candidate["rnpv_millions"] / total_rnpv
    else:
        concentration = 0.0

    return {
        "total_pipeline_rnpv": round(total_rnpv, 2),
        "candidate_count": len(candidates),
        "candidate_valuations": valuations,
        "concentration_risk": round(concentration, 4),
        "concentration_warning": concentration > 0.40,
    }


# ---------------------------------------------------------------------------
# Clinical Trial Signal Analysis
# ---------------------------------------------------------------------------

def analyze_trial_signals(
    candidate: DrugCandidate,
    competitor_count: int = 0,
    interim_result: Optional[str] = None,
    adverse_event_signal: bool = False,
) -> Dict[str, object]:
    """
    Generate investment signals from clinical trial intelligence.

    Parameters
    ----------
    candidate : DrugCandidate
        Drug candidate under analysis.
    competitor_count : int
        Number of competing trials in same indication.
    interim_result : str or None
        Interim analysis result: "positive", "neutral", "negative".
    adverse_event_signal : bool
        Whether a safety signal has been detected.

    Returns
    -------
    dict
        overall_signal, signal_components, investment_recommendation.
    """
    signals = {}

    # Enrollment velocity
    if candidate.enrollment_pct >= 1.0:
        signals["enrollment"] = ("POSITIVE", "Enrollment on/above target")
    elif candidate.enrollment_pct >= 0.75:
        signals["enrollment"] = ("NEUTRAL", "Enrollment slightly below target")
    else:
        signals["enrollment"] = ("NEGATIVE", "Enrollment significantly delayed")

    # Competitive landscape
    if competitor_count == 0:
        signals["competition"] = ("POSITIVE", "First-in-class advantage")
    elif competitor_count <= 2:
        signals["competition"] = ("NEUTRAL", f"{competitor_count} competitor(s)")
    else:
        signals["competition"] = (
            "NEGATIVE", f"Crowded field with {competitor_count} competitors"
        )

    # Interim analysis
    if interim_result:
        if interim_result.lower() == "positive":
            signals["interim"] = ("POSITIVE", "Positive interim results")
        elif interim_result.lower() == "negative":
            signals["interim"] = ("NEGATIVE", "Negative interim results")
        else:
            signals["interim"] = ("NEUTRAL", "Neutral interim results")

    # Safety signal
    if adverse_event_signal:
        signals["safety"] = ("NEGATIVE", "Adverse event safety signal detected")
    else:
        signals["safety"] = ("POSITIVE", "No safety signals")

    # Patent runway
    if candidate.patent_years_remaining >= 10:
        signals["patent"] = ("POSITIVE", f"{candidate.patent_years_remaining:.0f} years remaining")
    elif candidate.patent_years_remaining >= 5:
        signals["patent"] = ("NEUTRAL", f"{candidate.patent_years_remaining:.0f} years remaining")
    else:
        signals["patent"] = ("NEGATIVE", f"Patent cliff in {candidate.patent_years_remaining:.0f} years")

    # Overall signal
    positive = sum(1 for s, _ in signals.values() if s == "POSITIVE")
    negative = sum(1 for s, _ in signals.values() if s == "NEGATIVE")

    if negative >= 2 or adverse_event_signal:
        overall = "SELL"
    elif positive >= 3 and negative == 0:
        overall = "BUY"
    elif positive > negative:
        overall = "MODERATE BUY"
    else:
        overall = "HOLD"

    return {
        "overall_signal": overall,
        "signal_components": {
            k: {"signal": v[0], "detail": v[1]} for k, v in signals.items()
        },
        "positive_count": positive,
        "negative_count": negative,
    }


# ---------------------------------------------------------------------------
# Portfolio Optimisation (Mean-Variance)
# ---------------------------------------------------------------------------

def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    max_position_size: float = 0.15,
) -> Dict[str, object]:
    """
    Simple mean-variance portfolio optimisation for pharmaceutical equities.

    Uses analytical solution for the tangency portfolio (maximum Sharpe ratio).

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected annual returns for each asset.
    covariance_matrix : np.ndarray
        Covariance matrix of returns.
    risk_free_rate : float
        Risk-free rate.
    max_position_size : float
        Maximum single position size (for diversification).

    Returns
    -------
    dict
        weights, expected_return, expected_volatility, sharpe_ratio.
    """
    n = len(expected_returns)
    excess_returns = expected_returns - risk_free_rate

    # Analytical tangency portfolio
    try:
        inv_cov = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        # Fallback to equal weight if singular
        weights = np.ones(n) / n
    else:
        raw_weights = inv_cov @ excess_returns
        if raw_weights.sum() != 0:
            weights = raw_weights / raw_weights.sum()
        else:
            weights = np.ones(n) / n

    # Enforce max position size (adapted when n is small)
    effective_max = max(max_position_size, 1.0 / n)
    weights = np.clip(weights, 0, effective_max)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    # Portfolio metrics
    port_return = float(weights @ expected_returns)
    port_vol = float(np.sqrt(weights @ covariance_matrix @ weights))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return {
        "weights": weights.tolist(),
        "expected_return": round(port_return, 4),
        "expected_volatility": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_position": round(float(weights.max()), 4),
    }
