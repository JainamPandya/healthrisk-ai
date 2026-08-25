"""Tests for the pharmaceutical analytics module."""

import numpy as np
import pytest

from financial.pharma.pharma_analytics import (
    DrugCandidate,
    calculate_rnpv,
    value_pipeline,
    analyze_trial_signals,
    optimize_portfolio,
)


@pytest.fixture
def sample_candidate():
    return DrugCandidate(
        name="ABC-001",
        indication="oncology",
        current_phase="phase_3",
        peak_sales_estimate=2500.0,
        years_to_market=2.0,
        patent_years_remaining=11.0,
        enrollment_pct=1.05,
    )


class TestRNPV:
    def test_basic_rnpv(self, sample_candidate):
        result = calculate_rnpv(sample_candidate)
        assert result["rnpv_millions"] > 0
        assert 0 < result["cumulative_success_probability"] <= 1

    def test_marketed_drug(self):
        marketed = DrugCandidate(
            name="Established", indication="cardiovascular",
            current_phase="marketed", peak_sales_estimate=1000.0,
            years_to_market=0, patent_years_remaining=5.0,
        )
        result = calculate_rnpv(marketed)
        assert result["cumulative_success_probability"] == 1.0

    def test_earlier_phase_lower_probability(self):
        early = DrugCandidate(
            name="Early", indication="oncology",
            current_phase="phase_1", peak_sales_estimate=2500.0,
            years_to_market=8.0, patent_years_remaining=15.0,
        )
        late = DrugCandidate(
            name="Late", indication="oncology",
            current_phase="phase_3", peak_sales_estimate=2500.0,
            years_to_market=2.0, patent_years_remaining=11.0,
        )
        early_result = calculate_rnpv(early)
        late_result = calculate_rnpv(late)
        assert early_result["cumulative_success_probability"] < late_result["cumulative_success_probability"]


class TestPipelineValuation:
    def test_pipeline_value(self, sample_candidate):
        pipeline = [
            sample_candidate,
            DrugCandidate("ABC-002", "autoimmune", "phase_2", 500.0, 5.0, 12.0),
        ]
        result = value_pipeline(pipeline)
        assert result["total_pipeline_rnpv"] > 0
        assert result["candidate_count"] == 2
        assert len(result["candidate_valuations"]) == 2

    def test_concentration_risk(self, sample_candidate):
        tiny = DrugCandidate("Tiny", "rare_disease", "phase_1", 10.0, 10.0, 15.0)
        result = value_pipeline([sample_candidate, tiny])
        assert result["concentration_risk"] > 0.5


class TestTrialSignals:
    def test_positive_signals(self, sample_candidate):
        result = analyze_trial_signals(sample_candidate, competitor_count=0)
        assert result["positive_count"] > 0
        assert result["overall_signal"] in ["BUY", "MODERATE BUY", "HOLD", "SELL"]

    def test_safety_signal_triggers_sell(self, sample_candidate):
        result = analyze_trial_signals(
            sample_candidate, adverse_event_signal=True, competitor_count=5,
        )
        assert result["overall_signal"] == "SELL"
        assert result["negative_count"] >= 2


class TestPortfolioOptimisation:
    def test_basic_optimisation(self):
        returns = np.array([0.08, 0.12, 0.10, 0.06])
        cov = np.array([
            [0.04, 0.01, 0.02, 0.005],
            [0.01, 0.09, 0.03, 0.01],
            [0.02, 0.03, 0.06, 0.015],
            [0.005, 0.01, 0.015, 0.02],
        ])
        result = optimize_portfolio(returns, cov, max_position_size=0.35)
        assert abs(sum(result["weights"]) - 1.0) < 0.01
        assert result["sharpe_ratio"] > 0
        assert result["max_position"] <= 0.35 + 0.01  # allow floating point tolerance
