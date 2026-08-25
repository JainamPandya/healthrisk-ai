import pandas as pd

from healthrisk.predictor import HealthRiskPredictor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"
MODEL_FILE = "models/healthrisk_lightgbm.joblib"


print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("Training model...")

predictor = HealthRiskPredictor()

predictor.train(df)

predictor.save(
    MODEL_FILE
)

print("\nTraining complete.")