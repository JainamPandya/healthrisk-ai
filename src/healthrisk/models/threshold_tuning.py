import warnings
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from healthrisk.preprocessing import create_preprocessor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


def evaluate_thresholds(y_true, probabilities):
    """
    Evaluate multiple classification thresholds on a validation set.
    """
    results = []
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        results.append((threshold, precision, recall, f1))
    return results


def main():

    df = pd.read_csv(DATA_FILE, low_memory=False)

    X = df.drop(columns=["readmitted", "early_readmission"])
    y = df["early_readmission"]

    # Step 1: Split into Train+Val (80%) and Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # Step 2: Split Train+Val into Train (75% of 80% = 60%) and Val (25% of 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    preprocessor = create_preprocessor(df)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )

    print("Training LightGBM on training subset (60%)...")
    
    model.fit(X_train, y_train)
    
    # Extract Hyperparameters
    classifier = model.named_steps["classifier"]

    print("\n===== THRESHOLD TUNING (VALIDATION SET - 20%) =====")
    val_probabilities = model.predict_proba(X_val)[:, 1]
    
    results = evaluate_thresholds(y_val, val_probabilities)
    
    print("Threshold | Precision | Recall | F1")
    print("--------------------------------------")
    for threshold, precision, recall, f1 in results:
        print(f"{threshold:9.2f} | {precision:9.4f} | {recall:6.4f} | {f1:6.4f}")

    # Select the optimal threshold (e.g., maximizing F1 score)
    best_threshold = max(results, key=lambda x: x[3])[0]
    print(f"\nOptimal threshold selected on validation set: {best_threshold}")

    print("\n===== FINAL EVALUATION (UNTOUCHED TEST SET - 20%) =====")
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    final_precision = precision_score(y_test, test_predictions, zero_division=0)
    final_recall = recall_score(y_test, test_predictions, zero_division=0)
    final_f1 = f1_score(y_test, test_predictions, zero_division=0)

    print(f"Final Test Precision : {final_precision:.4f}")
    print(f"Final Test Recall    : {final_recall:.4f}")
    print(f"Final Test F1-Score  : {final_f1:.4f}")

if __name__ == "__main__":
    main()