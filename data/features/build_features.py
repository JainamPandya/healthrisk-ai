"""
Feature pipeline for HealthRisk AI.

Transforms processed patient data into feature matrices for model training
and evaluation, implementing:
- Clinical risk indices (comorbidity counts, lab trajectories)
- Polypharmacy scores
- Demographics and utilization encodings
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and build engineered clinical features from patient dataframe."""
    features = pd.DataFrame(index=df.index)

    # 1. Utilization intensity index
    if "number_inpatient" in df.columns and "number_emergency" in df.columns:
        features["utilization_intensity"] = (
            df["number_inpatient"] * 3.0 + df["number_emergency"] * 1.5 + df.get("number_outpatient", 0) * 0.5
        )

    # 2. Medication burden (polypharmacy proxy)
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
    ]
    existing_meds = [c for c in med_cols if c in df.columns]
    if existing_meds:
        # Count non-No medications
        active_meds = (df[existing_meds] != "No") & (df[existing_meds] != "nan")
        features["active_medication_count"] = active_meds.sum(axis=1)
        features["polypharmacy_flag"] = (features["active_medication_count"] >= 5).astype(int)

    # 3. Glycemic severity indicator
    if "A1Cresult" in df.columns:
        features["high_a1c"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

    if "max_glu_serum" in df.columns:
        features["high_glucose"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

    return features


if __name__ == "__main__":
    input_path = "data/processing/cleaned_diabetic_data.csv"
    if Path(input_path).exists():
        df = pd.read_csv(input_path, low_memory=False)
        feat = build_clinical_features(df)
        print(f"Generated {feat.shape[1]} engineered features for {len(feat)} encounters.")
