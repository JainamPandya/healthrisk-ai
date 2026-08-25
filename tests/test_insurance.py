"""Tests for the insurance actuarial module."""

import numpy as np
import pandas as pd
import pytest

from financial.insurance.actuarial import (
    calculate_hcc_risk_score,
    calculate_premium,
    estimate_ibnr_chain_ladder,
    calculate_mlr,
    stratify_members,
)


class TestHCCRiskScore:
    """Tests for HCC risk score calculation."""

    def test_basic_risk_score(self):
        result = calculate_hcc_risk_score(["E11", "I50"], age=65, gender="male")
        assert result["risk_score"] > 0
        assert result["hcc_count"] == 2
        assert "E11" in result["contributing_conditions"]
        assert "I50" in result["contributing_conditions"]

    def test_no_hcc_conditions(self):
        result = calculate_hcc_risk_score(["I10"], age=50, gender="female")
        assert result["hcc_count"] == 0
        assert result["condition_score"] == 0.0

    def test_demographic_base_included(self):
        result = calculate_hcc_risk_score([], age=65, gender="male")
        assert result["demographic_base"] > 0
        assert result["risk_score"] == result["demographic_base"]

    def test_empty_codes(self):
        result = calculate_hcc_risk_score([], age=30, gender="female")
        assert result["hcc_count"] == 0
        assert result["risk_score"] >= 0


class TestPremiumPricing:
    """Tests for premium calculation."""

    def test_basic_premium(self):
        result = calculate_premium(
            age=45, gender="male", diagnosis_codes=["E11"],
        )
        assert result.total_premium > 0
        assert result.base_rate == 450.0
        assert result.age_factor > 0

    def test_age_factor_increases_with_age(self):
        young = calculate_premium(age=25, gender="male", diagnosis_codes=[])
        old = calculate_premium(age=60, gender="male", diagnosis_codes=[])
        assert old.age_factor > young.age_factor
        assert old.total_premium > young.total_premium

    def test_clinical_risk_loading(self):
        healthy = calculate_premium(age=50, gender="male", diagnosis_codes=[])
        sick = calculate_premium(age=50, gender="male", diagnosis_codes=["E11", "I50", "N18"])
        assert sick.clinical_risk_loading > healthy.clinical_risk_loading
        assert sick.total_premium > healthy.total_premium


class TestIBNR:
    """Tests for IBNR reserve estimation."""

    def test_chain_ladder(self):
        triangle = np.array([
            [100, 150, 170, 180],
            [110, 160, 175, 0],
            [120, 170, 0,   0],
            [130, 0,   0,   0],
        ], dtype=float)
        result = estimate_ibnr_chain_ladder(triangle)
        assert result["estimated_ibnr"] >= 0
        assert len(result["development_factors"]) == 3
        assert result["total_ultimate"] >= result["total_paid"]


class TestMLR:
    """Tests for Medical Loss Ratio."""

    def test_compliant_mlr(self):
        result = calculate_mlr(850_000, 1_000_000)
        assert result["mlr"] == 0.85
        assert result["compliant_individual"] is True
        assert result["compliant_large_group"] is True

    def test_non_compliant_mlr(self):
        result = calculate_mlr(700_000, 1_000_000)
        assert result["mlr"] == 0.70
        assert result["compliant_individual"] is False
        assert result["rebate_required"] is True
        assert result["rebate_amount"] > 0

    def test_zero_premiums_raises(self):
        with pytest.raises(ValueError):
            calculate_mlr(100, 0)


class TestRiskStratification:
    """Tests for member risk stratification."""

    def test_stratification(self):
        df = pd.DataFrame({
            "member_id": [1, 2, 3, 4],
            "risk_score": [0.2, 0.7, 1.5, 3.0],
        })
        result = stratify_members(df)
        assert "risk_tier" in result.columns
        assert "care_program" in result.columns
        assert result.iloc[0]["risk_tier"] == "Tier 1: Low"
        assert result.iloc[3]["risk_tier"] == "Tier 4: Catastrophic"
