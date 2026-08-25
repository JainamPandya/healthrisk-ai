import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from healthrisk.preprocessing import create_preprocessor
from healthrisk.explainability import explain_model

from healthrisk.explainability import (
    explain_model,
    get_global_feature_importance,
)

DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


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

print("Calculating SHAP values...")

# Use a small sample first.
X_sample = X_test.iloc[:500]

explainer, shap_values, X_transformed = explain_model(
    model,
    X_sample,
)

print("\nSHAP calculation successful!")

print("Original features:", X_sample.shape[1])
print("Transformed features:", X_transformed.shape[1])
print("SHAP values shape:", shap_values.shape)


original_features = X.columns.tolist()

importance = get_global_feature_importance(
    model,
    X_sample,
    original_features,
)

print("\n===== TOP SHAP FEATURES =====")
print(importance.head(15).to_string(index=False))

print("\n===== SHAP DIRECTION =====")

for feature in [
    "number_inpatient",
    "discharge_disposition_id",
    "number_diagnoses",
    "time_in_hospital",
    "number_emergency",
]:
    matching = [
        i
        for i, name in enumerate(X_transformed.columns)
        if name.endswith(f"__{feature}")
        or f"__{feature}_" in name
    ]

    if matching:
        mean_shap = shap_values[:, matching].mean()

        print(
            f"{feature:30} "
            f"mean SHAP = {mean_shap:.6f}"
        )