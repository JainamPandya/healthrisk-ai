import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer
from healthrisk.preprocessing import create_preprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "race": ["Caucasian", "AfricanAmerican", "Caucasian"],
        "gender": ["Male", "Female", "Male"],
        "age": ["[50-60)", "[60-70)", "[70-80)"],
        "weight": ["?", "?", "?"],
        "admission_type_id": [1, 2, 1],
        "discharge_disposition_id": [1, 3, 1],
        "admission_source_id": [7, 7, 7],
        "time_in_hospital": [2, 10, 5],
        "payer_code": ["MC", "MC", "SP"],
        "medical_specialty": ["InternalMedicine", "InternalMedicine", "Cardiology"],
        "num_lab_procedures": [20, 60, 40],
        "num_procedures": [1, 4, 2],
        "num_medications": [5, 18, 12],
        "number_outpatient": [0, 2, 0],
        "number_emergency": [0, 3, 1],
        "number_inpatient": [0, 4, 1],
        "diag_1": ["486", "250.83", "414.01"],
        "diag_2": ["401.9", "250.01", "276"],
        "diag_3": ["250", "255", "428"],
        "number_diagnoses": [3, 9, 5],
        "max_glu_serum": ["None", ">300", "None"],
        "A1Cresult": ["None", ">8", "None"],
        "metformin": ["No", "Steady", "No"],
        "repaglinide": ["No", "No", "No"],
        "nateglinide": ["No", "No", "No"],
        "chlorpropamide": ["No", "No", "No"],
        "glimepiride": ["No", "No", "No"],
        "acetohexamide": ["No", "No", "No"],
        "glipizide": ["No", "No", "No"],
        "glyburide": ["No", "No", "No"],
        "tolbutamide": ["No", "No", "No"],
        "pioglitazone": ["No", "No", "No"],
        "rosiglitazone": ["No", "No", "No"],
        "acarbose": ["No", "No", "No"],
        "miglitol": ["No", "No", "No"],
        "troglitazone": ["No", "No", "No"],
        "tolazamide": ["No", "No", "No"],
        "examide": ["No", "No", "No"],
        "citoglipton": ["No", "No", "No"],
        "insulin": ["No", "Up", "No"],
        "glyburide-metformin": ["No", "No", "No"],
        "glipizide-metformin": ["No", "No", "No"],
        "glimepiride-pioglitazone": ["No", "No", "No"],
        "metformin-rosiglitazone": ["No", "No", "No"],
        "metformin-pioglitazone": ["No", "No", "No"],
        "change": ["No", "Ch", "No"],
        "diabetesMed": ["No", "Yes", "No"],
        "readmitted": ["NO", ">30", "NO"],
        "early_readmission": [0, 0, 0]
    })

def test_create_preprocessor(sample_data):
    preprocessor = create_preprocessor(sample_data)
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Check if transformers are correctly named and present
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "numerical" in transformer_names
    assert "categorical" in transformer_names

def test_preprocessing_transform(sample_data):
    preprocessor = create_preprocessor(sample_data)
    X = sample_data.drop(columns=["readmitted", "early_readmission"])
    
    # Fit and transform
    transformed_data = preprocessor.fit_transform(X)
    
    # Check that output is a numpy array and has more columns due to one-hot encoding
    assert hasattr(transformed_data, "shape")
    assert transformed_data.shape[0] == 3
    assert transformed_data.shape[1] > X.shape[1]
