import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from healthrisk.preprocessing import create_preprocessor
from healthrisk.patient_explanation import explain_patient

from healthrisk.visualization import plot_patient_explanation

DATA_FILE = "data/processing/cleaned_diabetic_data.csv"


df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

X = df.drop(
    columns=["readmitted", "early_readmission"]
)

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
        (
            "preprocessor",
            preprocessor,
        ),
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

model.fit(
    X_train,
    y_train,
)


# Select one patient
patient = X_test.iloc[[0]]


result = explain_patient(
    model,
    patient,
)


print("\n===== PATIENT RISK =====")

print(
    f"Risk Score: "
    f"{result['risk_score']:.4f}"
)

print(
    f"Risk Category: "
    f"{result['risk_category']}"
)


print(
    "\n===== FACTORS INCREASING RISK ====="
)

for _, row in result["factors_increasing"].iterrows():

    print(
        f"↑ {row['readable_feature']:<40} "
        f"{row['shap_value']:+.4f}"
    )


print(
    "\n===== FACTORS DECREASING RISK ====="
)

for _, row in result["factors_decreasing"].iterrows():

    print(
        f"↓ {row['readable_feature']:<40} "
        f"{row['shap_value']:+.4f}"
    )


print(
    "\nNOTE: The risk score is a model prediction, "
    "not a medical diagnosis."
)   

plot_patient_explanation(
    result,
    "reports/patient_risk_explanation.png",
)