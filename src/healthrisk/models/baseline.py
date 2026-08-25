import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from healthrisk.preprocessing import create_preprocessor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


def train_baseline():
    # Load data
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Separate features and target
    X = df.drop(columns=["readmitted", "early_readmission"])
    y = df["early_readmission"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create preprocessing
    preprocessor = create_preprocessor(df)

    # Create ML pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n===== BASELINE RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_baseline()