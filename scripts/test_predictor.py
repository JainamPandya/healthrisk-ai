import pandas as pd

from healthrisk.predictor import HealthRiskPredictor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("Training model...")

predictor = HealthRiskPredictor()

predictor.train(df)


# Select one patient
patient = df.drop(
    columns=["readmitted", "early_readmission"]
).iloc[[0]]


result = predictor.predict(
    patient
)


print("\n===== PREDICTION =====")

print(
    f"Risk Score: "
    f"{result['risk_score']:.4f}"
)

print(
    f"Risk Category: "
    f"{result['risk_category']}"
)