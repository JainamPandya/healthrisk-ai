import pandas as pd
import shap


def _clean_feature_name(feature_name):
    """
    Convert a transformed feature name into a readable name.
    """

    name = feature_name

    if "__" in name:
        name = name.split("__", 1)[1]

    readable_names = {
        "number_inpatient": "Number of inpatient visits",
        "number_emergency": "Number of emergency visits",
        "number_outpatient": "Number of outpatient visits",
        "num_lab_procedures": "Number of laboratory procedures",
        "num_medications": "Number of medications",
        "num_procedures": "Number of procedures",
        "number_diagnoses": "Number of diagnoses",
        "time_in_hospital": "Time in hospital",
    }

    if name in readable_names:
        return readable_names[name]

    encoded_features = {
        "discharge_disposition_id_": "Discharge disposition",
        "A1Cresult_": "A1C result",
        "payer_code_": "Payer code",
        "diabetesMed_": "Diabetes medication",
        "insulin_": "Insulin",
        "metformin_": "Metformin",
        "age_": "Age",
    }

    for prefix, readable_name in encoded_features.items():
        if name.startswith(prefix):
            return f"{readable_name} = {name[len(prefix):]}"

    if name.startswith("diag_"):
        diagnosis = name[len("diag_"):]
        if "_" in diagnosis:
            number, code = diagnosis.split("_", 1)
            return f"Diagnosis {number} = {code}"

    if "_" in name:
        prefix, value = name.split("_", 1)
        if prefix in ["race", "gender"]:
            return f"{prefix.title()} = {value}"

    return name


def explain_patient(model, patient):
    """
    Generate a patient-level SHAP explanation.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform patient
    patient_transformed = preprocessor.transform(patient)

    feature_names = preprocessor.get_feature_names_out()

    patient_transformed = pd.DataFrame(
        patient_transformed,
        columns=feature_names,
        index=patient.index,
    )

    # SHAP
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(
        patient_transformed
    )

    # SHAP 0.52 + LightGBM binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Model score
    risk_score = model.predict_proba(
        patient
    )[0, 1]

    # Risk category
    if risk_score < 0.20:
        risk_category = "Low"
    elif risk_score < 0.40:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    # SHAP values for this patient
    patient_shap = shap_values[0]

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": patient_shap,
            "absolute_shap": abs(patient_shap),
        }
    )

    # Sort by importance
    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False,
    )

    # Human-readable names
    explanation["readable_feature"] = (
        explanation["feature"]
        .apply(_clean_feature_name)
    )

    # Separate positive and negative factors
    factors_increasing = explanation[
        explanation["shap_value"] > 0
    ].head(5)

    factors_decreasing = explanation[
        explanation["shap_value"] < 0
    ].sort_values(
        "shap_value"
    ).head(5)

    return {
        "risk_score": float(risk_score),
        "risk_category": risk_category,
        "top_features": explanation.head(10),
        "factors_increasing": factors_increasing,
        "factors_decreasing": factors_decreasing,
    }