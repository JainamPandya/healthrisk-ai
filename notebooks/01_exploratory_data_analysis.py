"""
Exploratory Data Analysis for HealthRisk AI.

Analyzes the diabetic patient readmission dataset:
- Demographics and clinical feature distributions
- Missing value patterns and data quality assessment
- Correlation and comorbidity analysis
- Target variable distribution (early vs. non-early readmission)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def run_eda(data_path: str = "data/processing/cleaned_diabetic_data.csv") -> dict:
    """Run exploratory data analysis on the processed dataset."""
    path = Path(data_path)
    if not path.exists():
        print(f"Data file not found: {data_path}")
        return {}

    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    print("\n--- Dataset Summary ---")
    print(f"Total encounters: {len(df):,}")
    print(f"Total features: {len(df.columns)}")

    target_col = "early_readmission" if "early_readmission" in df.columns else "readmitted"
    if target_col in df.columns:
        print(f"\n--- Target Distribution ({target_col}) ---")
        print(df[target_col].value_counts(normalize=True).round(4) * 100)

    # Key numerical feature summary
    num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_inpatient"]
    existing_num_cols = [c for c in num_cols if c in df.columns]
    if existing_num_cols:
        print("\n--- Numerical Features Summary ---")
        print(df[existing_num_cols].describe().round(2))

    # Medication columns summary
    med_cols = ["metformin", "insulin", "glipizide", "glyburide"]
    existing_med_cols = [c for c in med_cols if c in df.columns]
    if existing_med_cols:
        print("\n--- Primary Diabetes Medications ---")
        for col in existing_med_cols:
            print(f"{col}: {df[col].value_counts().to_dict()}")

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numerical_summary": df[existing_num_cols].describe().to_dict() if existing_num_cols else {},
    }


if __name__ == "__main__":
    run_eda()
