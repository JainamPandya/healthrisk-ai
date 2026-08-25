import pandas as pd
from healthrisk.evaluation import print_evaluation
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from healthrisk.preprocessing import create_preprocessor


DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


def train_xgboost():

    print("Loading dataset...")

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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training XGBoost...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n===== XGBOOST RESULTS =====")

    results = print_evaluation(
        y_test,
        y_pred,
        y_probability,
    )


if __name__ == "__main__":
    train_xgboost()