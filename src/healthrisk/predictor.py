import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from healthrisk.preprocessing import create_preprocessor
from healthrisk.patient_explanation import explain_patient
from healthrisk.counterfactual import generate_counterfactual


class HealthRiskPredictor:

    def __init__(self):
        self.model = None

    def train(self, df):
        """
        Train the LightGBM health-risk model.
        """

        X = df.drop(
            columns=["readmitted", "early_readmission"]
        )

        y = df["early_readmission"]

        preprocessor = create_preprocessor(df)

        self.model = Pipeline(
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

        self.model.fit(X, y)

        return self

    def predict(self, patient):
        """
        Predict early-readmission risk for one patient.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        probability = self.model.predict_proba(
            patient
        )[0, 1]

        if probability < 0.20:
            risk_category = "Low"

        elif probability < 0.40:
            risk_category = "Moderate"

        else:
            risk_category = "High"

        return {
            "risk_score": float(probability),
            "risk_category": risk_category,
        }

    def explain(self, patient):
        """
        Generate SHAP explanation for one patient.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        return explain_patient(
            self.model,
            patient,
        )

    def counterfactual(
        self,
        patient,
        target_risk=0.20,
        max_changes=5,
    ):
        """
        Generate counterfactual explanation for one patient.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        return generate_counterfactual(
            self.model,
            patient,
            target_risk=target_risk,
            max_changes=max_changes,
        )

    def save(self, filepath):
        """
        Save trained model to disk.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        joblib.dump(
            self.model,
            filepath,
        )

        print(
            f"Model saved to: {filepath}"
        )

    def load(self, filepath):
        """
        Load trained model from disk.
        """

        self.model = joblib.load(
            filepath
        )

        print(
            f"Model loaded from: {filepath}"
        )

        return self