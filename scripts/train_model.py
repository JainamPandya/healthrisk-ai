import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from healthrisk.predictor import HealthRiskPredictor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"
MODEL_FILE = "models/healthrisk_lightgbm.joblib"


print("Loading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("Training model...")

mlflow.set_experiment("HealthRisk_Canonical_Training")
with mlflow.start_run() as run:
    # We will do a quick train/test split to log metrics
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["early_readmission"]
    )
    
    # Train the predictor on the training set
    predictor = HealthRiskPredictor()
    predictor.train(df_train)
    
    # Evaluate
    # Hardcode threshold 0.35 from previous threshold tuning, or just 0.5.
    # The default predictor uses probabilities and predicts categories.
    X_test = df_test.drop(columns=["readmitted", "early_readmission"])
    y_test = df_test["early_readmission"]
    
    # Evaluate
    # Hardcode threshold 0.35 from previous threshold tuning, or just 0.5.
    # The default predictor uses probabilities and predicts categories.
    model = predictor.model
    assert model is not None
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.35).astype(int)
    
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    # Log parameters
    classifier = model.named_steps["classifier"]
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", classifier.n_estimators)
    mlflow.log_param("max_depth", classifier.max_depth)
    mlflow.log_param("learning_rate", classifier.learning_rate)
    
    # Log metrics
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("f1_score", float(f1))
    
    print(f"Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Now train on full dataset for the final artifact
    print("Retraining on full dataset for final artifact...")
    predictor.train(df)
    
    # Save model
    predictor.save(MODEL_FILE)
    
    # Log artifacts
    mlflow.log_artifact(MODEL_FILE, "model_artifacts")
    
    # Tag run
    mlflow.set_tag("model_component", "canonical_lightgbm")
    mlflow.set_tag("data_version", "v1.0")

print("\nTraining complete.")