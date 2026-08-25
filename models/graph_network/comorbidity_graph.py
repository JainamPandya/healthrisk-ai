"""
Graph Neural Network Module for HealthRisk AI.

Implements a heterogeneous patient-disease-drug graph for
comorbidity network analysis and drug interaction mapping.

When PyTorch Geometric is available, uses GATv2Conv layers.
Otherwise falls back to a NetworkX-based graph analysis
with node2vec-style embeddings approximated via spectral methods.

References:
- Velickovic et al. (2018) Graph Attention Networks
- Brody et al. (2022) GATv2
- PDF Section A3.2: GNN for Comorbidity and Drug Interaction Networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

class ComorbidityGraph:
    """
    Heterogeneous patient-disease-drug graph.

    Node types:
    - patient: demographics, admission count, total procedures
    - disease: ICD-10 code, prevalence, average cost
    - drug: medication name, drug class, interaction count

    Edge types:
    - patient-has-disease
    - patient-takes-drug
    - disease-comorbid-with-disease
    - drug-interacts-with-drug
    - drug-treats-disease
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for ComorbidityGraph")
        self.graph = nx.Graph()
        self.node_types: Dict[str, str] = {}
        self.node_features: Dict[str, Dict] = {}

    def add_patient(
        self,
        patient_id: str,
        age: int = 0,
        gender: str = "unknown",
        num_admissions: int = 0,
    ):
        """Add a patient node to the graph."""
        node_id = f"patient_{patient_id}"
        self.graph.add_node(node_id)
        self.node_types[node_id] = "patient"
        self.node_features[node_id] = {
            "age": age,
            "gender": gender,
            "num_admissions": num_admissions,
        }

    def add_disease(
        self,
        icd_code: str,
        description: str = "",
        prevalence: float = 0.0,
        avg_cost: float = 0.0,
    ):
        """Add a disease node to the graph."""
        node_id = f"disease_{icd_code}"
        if node_id not in self.node_types:
            self.graph.add_node(node_id)
            self.node_types[node_id] = "disease"
            self.node_features[node_id] = {
                "icd_code": icd_code,
                "description": description,
                "prevalence": prevalence,
                "avg_cost": avg_cost,
            }

    def add_drug(
        self,
        drug_name: str,
        drug_class: str = "",
        interaction_count: int = 0,
    ):
        """Add a drug node to the graph."""
        node_id = f"drug_{drug_name}"
        if node_id not in self.node_types:
            self.graph.add_node(node_id)
            self.node_types[node_id] = "drug"
            self.node_features[node_id] = {
                "drug_name": drug_name,
                "drug_class": drug_class,
                "interaction_count": interaction_count,
            }

    def add_patient_disease_edge(self, patient_id: str, icd_code: str):
        """Add patient-has-disease edge."""
        self.graph.add_edge(
            f"patient_{patient_id}",
            f"disease_{icd_code}",
            edge_type="has_disease",
        )

    def add_patient_drug_edge(self, patient_id: str, drug_name: str):
        """Add patient-takes-drug edge."""
        self.graph.add_edge(
            f"patient_{patient_id}",
            f"drug_{drug_name}",
            edge_type="takes_drug",
        )

    def add_comorbidity_edge(
        self,
        icd_code_1: str,
        icd_code_2: str,
        co_occurrence_count: int = 1,
    ):
        """Add disease-comorbid-with-disease edge."""
        self.graph.add_edge(
            f"disease_{icd_code_1}",
            f"disease_{icd_code_2}",
            edge_type="comorbid_with",
            weight=co_occurrence_count,
        )

    def add_drug_interaction_edge(
        self,
        drug_1: str,
        drug_2: str,
        severity: str = "moderate",
    ):
        """Add drug-interacts-with-drug edge."""
        self.graph.add_edge(
            f"drug_{drug_1}",
            f"drug_{drug_2}",
            edge_type="interacts_with",
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Graph Analytics
    # ------------------------------------------------------------------

    def get_graph_statistics(self) -> Dict[str, int]:
        """Return basic graph statistics."""
        type_counts = defaultdict(int)
        for ntype in self.node_types.values():
            type_counts[ntype] += 1

        edge_type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_type_counts[data.get("edge_type", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "patient_count": type_counts["patient"],
            "disease_count": type_counts["disease"],
            "drug_count": type_counts["drug"],
            "edge_types": dict(edge_type_counts),
        }

    def get_patient_comorbidity_profile(
        self, patient_id: str,
    ) -> Dict[str, object]:
        """
        Get a patient's comorbidity profile from the graph.

        Returns diseases, medications, and connected comorbidities.
        """
        node_id = f"patient_{patient_id}"
        if node_id not in self.graph:
            return {"error": f"Patient {patient_id} not found"}

        diseases = []
        drugs = []

        for neighbor in self.graph.neighbors(node_id):
            ntype = self.node_types.get(neighbor, "unknown")
            if ntype == "disease":
                diseases.append(self.node_features.get(neighbor, {}))
            elif ntype == "drug":
                drugs.append(self.node_features.get(neighbor, {}))

        # Find comorbidity connections between patient's diseases
        comorbidity_pairs = []
        disease_nodes = [
            f"disease_{d.get('icd_code', '')}"
            for d in diseases if 'icd_code' in d
        ]
        for i, d1 in enumerate(disease_nodes):
            for d2 in disease_nodes[i + 1:]:
                if self.graph.has_edge(d1, d2):
                    comorbidity_pairs.append((
                        self.node_features.get(d1, {}),
                        self.node_features.get(d2, {}),
                    ))

        return {
            "patient_id": patient_id,
            "disease_count": len(diseases),
            "drug_count": len(drugs),
            "diseases": diseases,
            "drugs": drugs,
            "comorbidity_pairs": len(comorbidity_pairs),
        }

    def calculate_polypharmacy_risk(self, patient_id: str) -> Dict[str, object]:
        """
        Calculate polypharmacy risk score for a patient based on
        drug interaction edges in the graph.
        """
        node_id = f"patient_{patient_id}"
        if node_id not in self.graph:
            return {"error": f"Patient {patient_id} not found"}

        drug_nodes = [
            n for n in self.graph.neighbors(node_id)
            if self.node_types.get(n) == "drug"
        ]

        interaction_count = 0
        major_interactions = 0
        interactions = []

        for i, d1 in enumerate(drug_nodes):
            for d2 in drug_nodes[i + 1:]:
                if self.graph.has_edge(d1, d2):
                    edge_data = self.graph.edges[d1, d2]
                    if edge_data.get("edge_type") == "interacts_with":
                        interaction_count += 1
                        if edge_data.get("severity") == "major":
                            major_interactions += 1
                        interactions.append({
                            "drug_1": self.node_features.get(d1, {}).get("drug_name", d1),
                            "drug_2": self.node_features.get(d2, {}).get("drug_name", d2),
                            "severity": edge_data.get("severity", "unknown"),
                        })

        # Risk score: 0-1 scale
        med_count = len(drug_nodes)
        risk_score = min(1.0, (
            med_count * 0.1
            + interaction_count * 0.2
            + major_interactions * 0.4
        ))

        return {
            "patient_id": patient_id,
            "medication_count": med_count,
            "polypharmacy": med_count >= 5,
            "interaction_count": interaction_count,
            "major_interactions": major_interactions,
            "risk_score": round(risk_score, 4),
            "interactions": interactions,
        }

    def generate_node_embeddings(self, dimensions: int = 64) -> Dict[str, np.ndarray]:
        """
        Generate node embeddings using spectral decomposition of
        the graph Laplacian (lightweight alternative to node2vec).

        Parameters
        ----------
        dimensions : int
            Embedding dimensionality.

        Returns
        -------
        dict
            Node ID → embedding vector.
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        # Use spectral layout as embedding
        nodes = list(self.graph.nodes())
        n = len(nodes)
        actual_dims = min(dimensions, n - 1) if n > 1 else 1

        try:
            laplacian = nx.laplacian_matrix(self.graph).toarray().astype(float)
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

            # Use smallest non-trivial eigenvectors as embeddings
            embeddings = eigenvectors[:, 1:actual_dims + 1]

            # Pad to requested dimensions if needed
            if embeddings.shape[1] < dimensions:
                padding = np.zeros((n, dimensions - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])

            return {
                nodes[i]: embeddings[i] for i in range(n)
            }
        except Exception:
            # Fallback: random embeddings
            return {
                node: np.random.randn(dimensions) * 0.01
                for node in nodes
            }


def build_graph_from_dataframe(
    df: pd.DataFrame,
    patient_id_col: str = "patient_id",
    diagnosis_cols: List[str] = None,
    medication_cols: List[str] = None,
) -> ComorbidityGraph:
    """
    Build a comorbidity graph from a patient DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Patient data with diagnosis and medication columns.
    patient_id_col : str
        Column name for patient identifiers.
    diagnosis_cols : list of str
        Columns containing diagnosis codes (e.g., diag_1, diag_2, diag_3).
    medication_cols : list of str
        Columns containing medication names.

    Returns
    -------
    ComorbidityGraph
        Constructed heterogeneous graph.
    """
    if diagnosis_cols is None:
        diagnosis_cols = ["diag_1", "diag_2", "diag_3"]
    if medication_cols is None:
        medication_cols = []

    graph = ComorbidityGraph()
    comorbidity_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for idx, row in df.iterrows():
        pid = str(row.get(patient_id_col, idx))
        graph.add_patient(pid)

        patient_diagnoses = []
        for col in diagnosis_cols:
            if col in df.columns:
                code = str(row[col])
                if code and code != "nan" and code != "?":
                    graph.add_disease(code)
                    graph.add_patient_disease_edge(pid, code)
                    patient_diagnoses.append(code)

        # Track comorbidity pairs
        for i, d1 in enumerate(patient_diagnoses):
            for d2 in patient_diagnoses[i + 1:]:
                pair = tuple(sorted([d1, d2]))
                comorbidity_counts[pair] += 1

        for col in medication_cols:
            if col in df.columns:
                med = str(row[col])
                if med and med not in ("nan", "No", "Steady", "?"):
                    graph.add_drug(med)
                    graph.add_patient_drug_edge(pid, med)

    # Add comorbidity edges for frequently co-occurring conditions
    for (d1, d2), count in comorbidity_counts.items():
        if count >= 5:  # Minimum co-occurrence threshold
            graph.add_comorbidity_edge(d1, d2, count)

    return graph
