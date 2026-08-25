import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(df):
    """
    Create preprocessing pipeline for the HealthRisk AI dataset.
    """

    # Target columns are not features
    X = df.drop(
        columns=["readmitted", "early_readmission"]
    )

    # Categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # These are ID codes, but they represent categories,
    # not continuous numerical values.
    categorical_features += [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    # Numerical columns = everything that is not categorical
    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor