import matplotlib.pyplot as plt


def plot_patient_explanation(result, output_file):
    """
    Create a horizontal bar chart showing
    the factors influencing an individual patient's risk.
    """

    increasing = result["factors_increasing"].copy()
    decreasing = result["factors_decreasing"].copy()

    explanation = (
        __import__("pandas")
        .concat([increasing, decreasing])
        .sort_values("shap_value")
    )

    labels = explanation["readable_feature"]
    values = explanation["shap_value"]

    plt.figure(figsize=(10, 6))

    plt.barh(
        labels,
        values,
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel("SHAP value")
    plt.ylabel("Feature")
    plt.title(
        f"Patient Risk Explanation "
        f"(Score: {result['risk_score']:.3f})"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved explanation chart to: {output_file}"
    )