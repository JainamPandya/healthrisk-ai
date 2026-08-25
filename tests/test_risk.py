import pytest
from healthrisk.risk import classify_risk

def test_classify_risk_low():
    assert classify_risk(0.10) == "Low"
    assert classify_risk(0.19) == "Low"
    assert classify_risk(0.0) == "Low"

def test_classify_risk_moderate():
    assert classify_risk(0.20) == "Moderate"
    assert classify_risk(0.39) == "Moderate"

def test_classify_risk_high():
    assert classify_risk(0.40) == "High"
    assert classify_risk(0.80) == "High"
    assert classify_risk(1.0) == "High"
