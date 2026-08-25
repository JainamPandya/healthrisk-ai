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


def main():

    df = pd.read_csv(DATA_FILE, low_memory=False)

    X = df.drop(columns=["readmitted", "early_readmission"])
    y = df["early_readmission"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    print("Training LightGBM...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n===== THRESHOLD COMPARISON =====")
    print("Threshold | Precision | Recall | F1")
    print("--------------------------------------")

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )


if __name__ == "__main__":
    main()