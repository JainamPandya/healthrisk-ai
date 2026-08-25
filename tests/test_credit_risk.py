"""Tests for the hospital credit risk module."""

import pytest

from financial.credit_risk.hospital_credit import (
    HospitalFinancials,
    ClinicalQuality,
    score_financial_metrics,
    score_clinical_quality,
    calculate_credit_score,
    estimate_probability_of_default,
    detect_early_warnings,
)


@pytest.fixture
def sample_financials():
    return HospitalFinancials(
        name="Community Hospital",
        annual_revenue=450_000_000,
        operating_margin=0.018,
        dscr=1.95,
        days_cash_on_hand=85,
        total_debt=180_000_000,
        case_mix_index=1.42,
        bed_count=300,
        occupancy_rate=0.72,
        medicare_pct=0.48,
        medicaid_pct=0.22,
        commercial_pct=0.25,
    )


@pytest.fixture
def sample_quality():
    return ClinicalQuality(
        readmission_rate_30day=0.172,
        hcahps_overall_stars=3.0,
        hai_sir=1.2,
        cmi_trend=-0.04,
        ed_boarding_hours=8.2,
        surgical_volume_trend=-0.035,
    )


class TestFinancialScoring:
    def test_score_financial_metrics(self, sample_financials):
        scores = score_financial_metrics(sample_financials)
        assert all(0 <= v <= 100 for v in scores.values())
        assert "operating_margin" in scores
        assert "dscr" in scores

    def test_high_margin_scores_high(self):
        financials = HospitalFinancials(
            name="Strong Hospital", annual_revenue=1_000_000_000,
            operating_margin=0.08, dscr=3.5, days_cash_on_hand=250,
            total_debt=100_000_000, case_mix_index=1.8, bed_count=500,
            occupancy_rate=0.80, medicare_pct=0.30, medicaid_pct=0.10,
            commercial_pct=0.55,
        )
        scores = score_financial_metrics(financials)
        assert scores["operating_margin"] == 100
        assert scores["dscr"] == 100


class TestClinicalScoring:
    def test_score_clinical_quality(self, sample_quality):
        scores = score_clinical_quality(sample_quality)
        assert all(0 <= v <= 100 for v in scores.values())
        assert "readmission" in scores

    def test_high_readmission_scores_low(self, sample_quality):
        scores = score_clinical_quality(sample_quality)
        assert scores["readmission"] <= 50


class TestCreditScore:
    def test_composite_score(self, sample_financials, sample_quality):
        result = calculate_credit_score(sample_financials, sample_quality)
        assert 0 <= result["composite_score"] <= 100
        assert result["implied_rating"] in ["AA", "A", "BBB+", "BBB", "BBB-", "BB+", "BB", "B"]

    def test_financial_weight(self, sample_financials, sample_quality):
        result = calculate_credit_score(
            sample_financials, sample_quality,
            financial_weight=1.0, clinical_weight=0.0,
        )
        assert result["clinical_score"] >= 0


class TestProbabilityOfDefault:
    def test_pd_range(self):
        result = estimate_probability_of_default(50.0, years=1)
        assert 0 <= result["pd_1yr"] <= 1
        assert 0 <= result["pd_cumulative"] <= 1

    def test_higher_score_lower_pd(self):
        low = estimate_probability_of_default(30.0)
        high = estimate_probability_of_default(80.0)
        assert high["pd_1yr"] < low["pd_1yr"]

    def test_multi_year_pd(self):
        result_1yr = estimate_probability_of_default(50.0, years=1)
        result_5yr = estimate_probability_of_default(50.0, years=5)
        assert result_5yr["pd_cumulative"] >= result_1yr["pd_cumulative"]


class TestEarlyWarnings:
    def test_detects_warnings(self, sample_financials, sample_quality):
        warnings = detect_early_warnings(sample_quality, sample_financials)
        assert len(warnings) > 0
        assert any(w["indicator"] == "Elevated Readmission Rate" for w in warnings)

    def test_no_warnings_for_good_hospital(self, sample_financials):
        good_quality = ClinicalQuality(
            readmission_rate_30day=0.12,
            hcahps_overall_stars=4.5,
            hai_sir=0.5,
            cmi_trend=0.02,
            ed_boarding_hours=1.5,
            surgical_volume_trend=0.05,
        )
        sample_financials.days_cash_on_hand = 200
        warnings = detect_early_warnings(good_quality, sample_financials)
        assert len(warnings) == 0
