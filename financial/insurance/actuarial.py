"""
Insurance Actuarial Module for HealthRisk AI.

Implements GLM-based premium pricing, IBNR reserve estimation,
and clinical risk stratification for health insurance actuarial desks.

References:
- CMS HCC Risk Adjustment Methodology
- ACA Medical Loss Ratio Requirements
- Bornhuetter-Ferguson IBNR estimation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# HCC Risk Score Calculator
# ---------------------------------------------------------------------------

# Simplified HCC mapping from common ICD-10 chapters to risk weights
# In production, this would use the full CMS-HCC model (V28)
HCC_WEIGHTS: Dict[str, float] = {
    "E11": 0.302,   # Type 2 diabetes without complication
    "E13": 0.302,   # Other specified diabetes
    "I10": 0.0,     # Essential hypertension (not HCC-relevant)
    "I50": 0.368,   # Heart failure
    "N18": 0.069,   # Chronic kidney disease
    "J44": 0.328,   # COPD
    "C34": 2.484,   # Malignant neoplasm of lung
    "F32": 0.309,   # Major depressive disorder
    "G20": 0.606,   # Parkinson's disease
    "I25": 0.140,   # Chronic ischemic heart disease
}

DEMOGRAPHIC_BASE_RATES: Dict[str, float] = {
    "male_65_69": 0.421,
    "male_70_74": 0.535,
    "female_65_69": 0.370,
    "female_70_74": 0.480,
    "default": 0.300,
}


def calculate_hcc_risk_score(
    diagnosis_codes: List[str],
    age: int = 65,
    gender: str = "male",
) -> Dict[str, float]:
    """
    Calculate CMS-HCC risk score from diagnosis codes.

    Parameters
    ----------
    diagnosis_codes : list of str
        ICD-10 diagnosis codes (e.g., ["E11", "I50", "N18"]).
    age : int
        Patient age.
    gender : str
        Patient gender ("male" or "female").

    Returns
    -------
    dict
        risk_score, hcc_count, contributing_conditions, demographic_base.
    """
    # Demographic base rate
    age_band = f"{age // 5 * 5}_{age // 5 * 5 + 4}"
    demo_key = f"{gender.lower()}_{age_band}"
    demographic_base = DEMOGRAPHIC_BASE_RATES.get(
        demo_key, DEMOGRAPHIC_BASE_RATES["default"]
    )

    # HCC condition contributions
    hcc_contributions = {}
    for code in diagnosis_codes:
        prefix = code[:3]
        if prefix in HCC_WEIGHTS and HCC_WEIGHTS[prefix] > 0:
            hcc_contributions[prefix] = HCC_WEIGHTS[prefix]

    condition_score = sum(hcc_contributions.values())
    total_score = demographic_base + condition_score

    return {
        "risk_score": round(total_score, 4),
        "hcc_count": len(hcc_contributions),
        "demographic_base": round(demographic_base, 4),
        "condition_score": round(condition_score, 4),
        "contributing_conditions": hcc_contributions,
    }


# ---------------------------------------------------------------------------
# GLM-Based Premium Pricing
# ---------------------------------------------------------------------------

@dataclass
class PremiumComponents:
    """Decomposed health insurance premium."""
    base_rate: float
    age_factor: float
    clinical_risk_loading: float
    geographic_factor: float
    administrative_loading: float
    profit_margin: float
    total_premium: float


def calculate_premium(
    age: int,
    gender: str,
    diagnosis_codes: List[str],
    region: str = "southeast",
    plan_type: str = "standard",
    base_monthly_rate: float = 450.0,
) -> PremiumComponents:
    """
    Calculate risk-adjusted health insurance premium using a
    simplified GLM approach.

    Premium = Base Rate × Age Factor × Clinical Risk Loading
              + Administrative Loading + Profit Margin

    Parameters
    ----------
    age : int
        Member age.
    gender : str
        Member gender.
    diagnosis_codes : list of str
        Known ICD-10 diagnosis codes.
    region : str
        Geographic region.
    plan_type : str
        Insurance plan type.
    base_monthly_rate : float
        Base community rate.

    Returns
    -------
    PremiumComponents
        Decomposed premium with each component.
    """
    # Age rating factor (ACA 3:1 band)
    if age < 21:
        age_factor = 0.635
    elif age < 30:
        age_factor = 0.850
    elif age < 40:
        age_factor = 1.000
    elif age < 50:
        age_factor = 1.200
    elif age < 60:
        age_factor = 1.550
    else:
        age_factor = min(1.900, 1.550 + (age - 60) * 0.035)

    # Clinical risk loading from HCC score
    hcc = calculate_hcc_risk_score(diagnosis_codes, age, gender)
    clinical_risk_loading = hcc["risk_score"] * 120.0  # $120 per HCC unit

    # Geographic factor
    geo_factors = {
        "northeast": 1.15, "southeast": 1.00,
        "midwest": 0.95, "west": 1.10, "default": 1.00,
    }
    geographic_factor = geo_factors.get(region, geo_factors["default"])

    # Calculate components
    risk_premium = base_monthly_rate * age_factor * geographic_factor
    administrative_loading = risk_premium * 0.12  # 12% admin
    profit_margin = risk_premium * 0.03           # 3% profit

    total = risk_premium + clinical_risk_loading + administrative_loading + profit_margin

    return PremiumComponents(
        base_rate=round(base_monthly_rate, 2),
        age_factor=round(age_factor, 4),
        clinical_risk_loading=round(clinical_risk_loading, 2),
        geographic_factor=round(geographic_factor, 4),
        administrative_loading=round(administrative_loading, 2),
        profit_margin=round(profit_margin, 2),
        total_premium=round(total, 2),
    )


# ---------------------------------------------------------------------------
# IBNR Reserve Estimation (Chain Ladder + Bornhuetter-Ferguson)
# ---------------------------------------------------------------------------

def estimate_ibnr_chain_ladder(
    claims_triangle: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate IBNR reserves using the Chain Ladder method.

    Parameters
    ----------
    claims_triangle : np.ndarray
        Upper-left cumulative claims development triangle.
        Shape: (n_origin_periods, n_development_periods).

    Returns
    -------
    dict
        estimated_ibnr, development_factors, ultimate_claims.
    """
    n_rows, n_cols = claims_triangle.shape
    development_factors = []

    for j in range(n_cols - 1):
        numerator = 0.0
        denominator = 0.0
        for i in range(n_rows - j - 1):
            if claims_triangle[i, j] > 0:
                numerator += claims_triangle[i, j + 1]
                denominator += claims_triangle[i, j]
        if denominator > 0:
            development_factors.append(numerator / denominator)
        else:
            development_factors.append(1.0)

    # Project ultimate claims for each origin period
    ultimate_claims = []
    for i in range(n_rows):
        latest_diagonal = n_cols - i - 1
        if latest_diagonal < 0:
            latest_diagonal = 0
        projected = claims_triangle[i, min(latest_diagonal, n_cols - 1)]
        for j in range(latest_diagonal, n_cols - 1):
            if j < len(development_factors):
                projected *= development_factors[j]
        ultimate_claims.append(projected)

    # IBNR = Ultimate - Paid-to-date
    total_paid = sum(
        claims_triangle[i, min(n_cols - i - 1, n_cols - 1)]
        for i in range(n_rows)
    )
    total_ultimate = sum(ultimate_claims)
    estimated_ibnr = total_ultimate - total_paid

    return {
        "estimated_ibnr": round(estimated_ibnr, 2),
        "total_ultimate": round(total_ultimate, 2),
        "total_paid": round(total_paid, 2),
        "development_factors": [round(f, 4) for f in development_factors],
    }


# ---------------------------------------------------------------------------
# Medical Loss Ratio (MLR) Calculator
# ---------------------------------------------------------------------------

def calculate_mlr(
    incurred_claims: float,
    earned_premiums: float,
    quality_improvement_expenses: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate Medical Loss Ratio per ACA requirements.

    MLR = (Incurred Claims + Quality Improvement) / Earned Premiums

    ACA requires MLR >= 80% for individual/small group,
    >= 85% for large group.

    Parameters
    ----------
    incurred_claims : float
        Total incurred medical claims.
    earned_premiums : float
        Total earned premiums.
    quality_improvement_expenses : float
        Quality improvement activity expenses.

    Returns
    -------
    dict
        mlr, compliant_individual, compliant_large_group, rebate_required.
    """
    if earned_premiums <= 0:
        raise ValueError("Earned premiums must be positive")

    mlr = (incurred_claims + quality_improvement_expenses) / earned_premiums

    return {
        "mlr": round(mlr, 4),
        "mlr_percentage": round(mlr * 100, 2),
        "compliant_individual": mlr >= 0.80,
        "compliant_large_group": mlr >= 0.85,
        "rebate_required": mlr < 0.80,
        "rebate_amount": round(
            max(0, (0.80 - mlr) * earned_premiums), 2
        ) if mlr < 0.80 else 0.0,
    }


# ---------------------------------------------------------------------------
# Member Risk Stratification
# ---------------------------------------------------------------------------

def stratify_members(
    members_df: pd.DataFrame,
    risk_score_column: str = "risk_score",
) -> pd.DataFrame:
    """
    Stratify insurance members into risk tiers based on clinical
    trajectory scores for care management programs.

    Parameters
    ----------
    members_df : pd.DataFrame
        DataFrame with member data including risk scores.
    risk_score_column : str
        Column name containing risk scores.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'risk_tier' and 'care_program' columns.
    """
    df = members_df.copy()

    conditions = [
        df[risk_score_column] >= 2.0,
        df[risk_score_column] >= 1.0,
        df[risk_score_column] >= 0.5,
    ]
    choices = ["Tier 4: Catastrophic", "Tier 3: High", "Tier 2: Moderate"]
    df["risk_tier"] = np.select(conditions, choices, default="Tier 1: Low")

    care_programs = {
        "Tier 4: Catastrophic": "Intensive Case Management",
        "Tier 3: High": "Disease Management Program",
        "Tier 2: Moderate": "Health Coaching",
        "Tier 1: Low": "Preventive Wellness",
    }
    df["care_program"] = df["risk_tier"].map(care_programs)

    return df
