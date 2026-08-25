"""Tests for the survival analysis module."""

import numpy as np
import pandas as pd
import pytest

from models.survival.survival_analysis import (
    kaplan_meier_estimate,
    prepare_readmission_survival_data,
)


class TestKaplanMeier:
    def test_basic_km(self):
        durations = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        events = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        result = kaplan_meier_estimate(durations, events, label="Test")
        assert result["label"] == "Test"
        assert result["n_events"] == 5
        assert result["n_censored"] == 3
        assert len(result["survival_table"]) > 0

    def test_all_events(self):
        durations = np.array([1, 2, 3, 4, 5])
        events = np.ones(5)
        result = kaplan_meier_estimate(durations, events)
        assert result["n_events"] == 5
        assert result["n_censored"] == 0

    def test_all_censored(self):
        durations = np.array([1, 2, 3, 4, 5])
        events = np.zeros(5)
        result = kaplan_meier_estimate(durations, events)
        assert result["n_events"] == 0
        assert result["n_censored"] == 5


class TestSurvivalDataPreparation:
    def test_prepare_data(self):
        df = pd.DataFrame({
            "time_in_hospital": [3, 5, 7, 2, 10],
            "early_readmission": [1, 0, 1, 0, 1],
            "num_medications": [10, 5, 15, 3, 20],
            "number_inpatient": [2, 0, 3, 0, 5],
        })
        result = prepare_readmission_survival_data(df)
        assert "duration" in result.columns
        assert "event" in result.columns
        assert len(result) == 5
        assert (result["duration"] > 0).all()

    def test_missing_time_column(self):
        df = pd.DataFrame({
            "early_readmission": [1, 0, 1],
            "num_medications": [10, 5, 15],
        })
        result = prepare_readmission_survival_data(df)
        assert "duration" in result.columns
        assert (result["duration"] > 0).all()
