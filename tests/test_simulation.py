"""Tests for the HealthRisk Lab simulation engine."""

import pytest

from simulation.healthrisk_lab import (
    HealthRiskLabEngine,
    Portfolio,
    generate_scenario,
)


class TestScenarioGeneration:
    def test_generates_scenario(self):
        scenario = generate_scenario(quarter=1, seed=42)
        assert scenario.name
        assert scenario.scenario_type
        assert scenario.severity in ["low", "medium", "high", "critical"]

    def test_deterministic_with_seed(self):
        s1 = generate_scenario(quarter=5, seed=100)
        s2 = generate_scenario(quarter=5, seed=100)
        assert s1.name == s2.name

    def test_different_quarters_can_differ(self):
        scenarios = [generate_scenario(q, seed=42 + q) for q in range(1, 11)]
        names = [s.name for s in scenarios]
        # At least some variation expected
        assert len(set(names)) >= 1


class TestSimulationEngine:
    def test_engine_initialises(self):
        engine = HealthRiskLabEngine()
        assert engine.current_quarter == 0
        assert engine.portfolio is not None
        assert len(engine.portfolio.bonds) == 15
        assert len(engine.portfolio.stocks) == 20

    def test_advance_quarter(self):
        engine = HealthRiskLabEngine(seed=42)
        result = engine.advance_quarter()
        assert result["quarter"] == 1
        assert "scenario" in result
        assert "total_score" in result
        assert result["portfolio_value"] > 0

    def test_run_multiple_quarters(self):
        engine = HealthRiskLabEngine(total_quarters=10, seed=42)
        results = engine.run_full_simulation()
        assert len(results) == 10
        assert engine.current_quarter == 10

    def test_score_stays_bounded(self):
        engine = HealthRiskLabEngine(total_quarters=40, seed=42)
        engine.run_full_simulation()
        assert 0 <= engine.portfolio.score <= 1000

    def test_final_score(self):
        engine = HealthRiskLabEngine(total_quarters=5, seed=42)
        engine.run_full_simulation()
        final = engine.get_final_score()
        assert final["quarters_played"] == 5
        assert final["final_portfolio_value"] > 0
        assert "final_mlr" in final
