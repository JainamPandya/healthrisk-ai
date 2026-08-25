import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/acquisition/diabetic_data.csv")
OUTPUT_FILE = Path("data/processing/cleaned_diabetic_data.csv")


def clean_data():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")

    # Convert '?' to proper missing values
    df = df.replace("?", pd.NA)

    # Create binary target:
    # 1 = readmitted within 30 days
    # 0 = not readmitted within 30 days
    df["early_readmission"] = (df["readmitted"] == "<30").astype(int)

    # Remove identifier columns
    df = df.drop(columns=["encounter_id", "patient_nbr"])

    # Remove columns with only one unique value
    df = df.drop(columns=["examide", "citoglipton"])

    print(f"Cleaned shape: {df.shape}")

    print("\nTarget distribution:")
    print(df["early_readmission"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_data()