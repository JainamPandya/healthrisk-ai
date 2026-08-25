"""Tests for the clinical NLP module."""

import pytest

from models.clinical_nlp.clinical_nlp import (
    preprocess_clinical_text,
    extract_clinical_entities,
    ClinicalTextClassifier,
    score_clinical_complexity,
)


class TestTextPreprocessing:
    def test_lowercases(self):
        result = preprocess_clinical_text("PATIENT ADMITTED")
        assert result == "patient admitted"

    def test_expands_abbreviations(self):
        result = preprocess_clinical_text("pt c/o chest pain")
        assert "patient" in result
        assert "complaining of" in result

    def test_normalises_whitespace(self):
        result = preprocess_clinical_text("too   many    spaces")
        assert "  " not in result


class TestNER:
    def test_extracts_medications(self):
        text = "Patient on metformin 1000mg and insulin for diabetes."
        entities = extract_clinical_entities(text)
        assert "metformin" in entities["medications"]
        assert "insulin" in entities["medications"]

    def test_extracts_conditions(self):
        text = "History of diabetes and hypertension."
        entities = extract_clinical_entities(text)
        assert "diabetes" in entities["conditions"]
        assert "hypertension" in entities["conditions"]

    def test_extracts_procedures(self):
        text = "Patient underwent cardiac catheterization and CT scan."
        entities = extract_clinical_entities(text)
        assert "cardiac catheterization" in entities["procedures"]
        assert "ct scan" in entities["procedures"]

    def test_empty_text(self):
        entities = extract_clinical_entities("")
        assert entities["medications"] == []
        assert entities["conditions"] == []
        assert entities["procedures"] == []


class TestTextClassifier:
    def test_tfidf_lr_fit_predict(self):
        texts = [
            "Patient with diabetes and hypertension, multiple medications",
            "Healthy young patient, routine checkup, no complaints",
            "Heart failure, renal failure, ICU admission, ventilator",
            "Mild cough, otherwise healthy, discharged same day",
        ] * 5  # Repeat for minimum training size
        labels = ["high_risk", "low_risk", "high_risk", "low_risk"] * 5

        clf = ClinicalTextClassifier(model_name="tfidf_lr")
        clf.fit(texts, labels)
        predictions = clf.predict(texts[:2])
        assert len(predictions) == 2
        assert all(p in ["high_risk", "low_risk"] for p in predictions)

    def test_predict_proba(self):
        texts = ["diabetes hypertension medications"] * 10
        labels = ["high_risk"] * 5 + ["low_risk"] * 5

        clf = ClinicalTextClassifier(model_name="tfidf_lr")
        clf.fit(texts + ["healthy checkup normal"] * 10, labels + ["low_risk"] * 10)
        proba = clf.predict_proba(["diabetes heart failure"])
        assert proba.shape[1] == 2  # binary classification


class TestComplexityScorer:
    def test_complex_note(self):
        text = (
            "Patient with diabetes, hypertension, and COPD. "
            "On metformin, insulin, lisinopril, and furosemide. "
            "Underwent cardiac catheterization and CT scan. "
            "Multiple comorbidities with complex medication regimen."
        )
        result = score_clinical_complexity(text)
        assert result["complexity_score"] > 0
        assert result["medication_count"] > 0
        assert result["condition_count"] > 0

    def test_simple_note(self):
        text = "Routine follow up. No complaints."
        result = score_clinical_complexity(text)
        assert result["complexity_score"] < 0.5
        assert result["medication_count"] == 0
