import pandas as pd
import shap


def explain_model(model, X):
    """
    Generate SHAP values for a trained LightGBM pipeline.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_transformed)

    # SHAP 0.52 + LightGBM can return a list for binary classification.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values, X_transformed


def get_global_feature_importance(
    model,
    X,
    original_features,
):
    """
    Calculate global SHAP importance for the original features.
    """

    _, shap_values, X_transformed = explain_model(
        model,
        X,
    )

    transformed_names = X_transformed.columns

    importance = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": abs(shap_values).mean(axis=0),
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    results = []

    for original_feature in original_features:

        matching = [
            i
            for i, name in enumerate(transformed_names)
            if name.endswith(f"__{original_feature}")
            or f"__{original_feature}_" in name
        ]

        if matching:

            total_importance = abs(
                shap_values[:, matching]
            ).mean()

            results.append(
                {
                    "feature": original_feature,
                    "importance": total_importance,
                }
            )

    return pd.DataFrame(results).sort_values(
        "importance",
        ascending=False,
    )