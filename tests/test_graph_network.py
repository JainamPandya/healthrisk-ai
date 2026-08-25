"""Tests for the Graph Neural Network (comorbidity graph) module."""

import pytest

from models.graph_network.comorbidity_graph import (
    ComorbidityGraph,
)


@pytest.fixture
def sample_graph():
    g = ComorbidityGraph()
    # Add patients
    g.add_patient("P1", age=65, gender="male", num_admissions=3)
    g.add_patient("P2", age=72, gender="female", num_admissions=1)

    # Add diseases
    g.add_disease("E11", "Type 2 Diabetes", 0.10, 15000)
    g.add_disease("I50", "Heart Failure", 0.05, 25000)
    g.add_disease("N18", "CKD", 0.08, 20000)

    # Add drugs
    g.add_drug("metformin", "biguanide", 2)
    g.add_drug("insulin", "insulin", 3)
    g.add_drug("lisinopril", "ace_inhibitor", 1)

    # Add edges
    g.add_patient_disease_edge("P1", "E11")
    g.add_patient_disease_edge("P1", "I50")
    g.add_patient_disease_edge("P2", "E11")
    g.add_patient_disease_edge("P2", "N18")

    g.add_patient_drug_edge("P1", "metformin")
    g.add_patient_drug_edge("P1", "insulin")
    g.add_patient_drug_edge("P1", "lisinopril")
    g.add_patient_drug_edge("P2", "metformin")

    g.add_comorbidity_edge("E11", "I50", 150)
    g.add_comorbidity_edge("E11", "N18", 200)

    g.add_drug_interaction_edge("metformin", "lisinopril", "moderate")

    return g


class TestGraphConstruction:
    def test_graph_statistics(self, sample_graph):
        stats = sample_graph.get_graph_statistics()
        assert stats["patient_count"] == 2
        assert stats["disease_count"] == 3
        assert stats["drug_count"] == 3
        assert stats["total_nodes"] == 8
        assert stats["total_edges"] > 0

    def test_duplicate_nodes_prevented(self, sample_graph):
        sample_graph.add_disease("E11", "Type 2 Diabetes Again")
        stats = sample_graph.get_graph_statistics()
        assert stats["disease_count"] == 3  # Should not increase


class TestPatientProfile:
    def test_comorbidity_profile(self, sample_graph):
        profile = sample_graph.get_patient_comorbidity_profile("P1")
        assert profile["disease_count"] == 2
        assert profile["drug_count"] == 3

    def test_unknown_patient(self, sample_graph):
        profile = sample_graph.get_patient_comorbidity_profile("UNKNOWN")
        assert "error" in profile


class TestPolypharmacy:
    def test_polypharmacy_risk(self, sample_graph):
        risk = sample_graph.calculate_polypharmacy_risk("P1")
        assert risk["medication_count"] == 3
        assert risk["interaction_count"] >= 1
        assert risk["risk_score"] > 0

    def test_low_polypharmacy(self, sample_graph):
        risk = sample_graph.calculate_polypharmacy_risk("P2")
        assert risk["medication_count"] == 1
        assert risk["polypharmacy"] is False


class TestNodeEmbeddings:
    def test_embeddings_generated(self, sample_graph):
        embeddings = sample_graph.generate_node_embeddings(dimensions=16)
        assert len(embeddings) == 8
        for node, emb in embeddings.items():
            assert len(emb) == 16
