"""
Partial Dependence Plot (PDP) analysis for HealthRisk AI.

Generates global partial dependence plots for numerical features
using the trained LightGBM sklearn Pipeline. PDPs show how the
predicted probability of early readmission changes as a single
feature varies, averaged across the dataset.

Uses sklearn.inspection.partial_dependence and matplotlib.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from healthrisk.config import MODELS_DIR, REPORTS_DIR


# Default features to plot — the most important numerical features.
DEFAULT_PDP_FEATURES = [
    "number_inpatient",
    "number_diagnoses",
    "time_in_hospital",
    "number_emergency",
    "num_lab_procedures",
    "num_medications",
]

# Human-readable display names for features.
FEATURE_DISPLAY_NAMES = {
    "number_inpatient": "Number of Inpatient Visits",
    "number_diagnoses": "Number of Diagnoses",
    "time_in_hospital": "Time in Hospital (days)",
    "number_emergency": "Number of Emergency Visits",
    "num_lab_procedures": "Number of Lab Procedures",
    "num_medications": "Number of Medications",
    "num_procedures": "Number of Procedures",
    "number_outpatient": "Number of Outpatient Visits",
}

# Output directory for PDP figures.
PDP_DIR = REPORTS_DIR / "pdp"


def load_model(model_path=None):
    """
    Load the trained sklearn Pipeline from disk.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the joblib model file. Defaults to
        models/healthrisk_lightgbm.joblib.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted preprocessing + classifier pipeline.
    """

    if model_path is None:
        model_path = MODELS_DIR / "healthrisk_lightgbm.joblib"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def get_model_features(model):
    """
    Extract the list of original feature names that the fitted
    preprocessing pipeline expects.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.

    Returns
    -------
    list of str
        Feature names from the preprocessor's transformers.
    """

    preprocessor = model.named_steps["preprocessor"]
    features = []

    for name, _transformer, columns in preprocessor.transformers_:
        features.extend(list(columns))

    return features


def validate_features(model, features):
    """
    Validate that all requested features exist in the model's
    input feature set.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.
    features : list of str
        Feature names to validate.

    Returns
    -------
    list of str
        Valid features (subset of ``features``).

    Raises
    ------
    ValueError
        If none of the requested features are valid.
    """

    model_features = get_model_features(model)
    valid = [f for f in features if f in model_features]
    invalid = [f for f in features if f not in model_features]

    if invalid:
        print(
            f"Warning: features not found in model "
            f"and will be skipped: {invalid}"
        )

    if not valid:
        raise ValueError(
            f"None of the requested features are valid. "
            f"Available features: {model_features}"
        )

    return valid


def compute_pdp(model, X, feature, grid_resolution=50):
    """
    Compute partial dependence for a single feature.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline.
    X : pd.DataFrame
        Sample of input data (raw features, same format as
        training data without target columns).
    feature : str
        The feature name to compute PDP for.
    grid_resolution : int, optional
        Number of grid points. Default 50.

    Returns
    -------
    dict
        ``values`` (1-D array of feature grid values) and
        ``avg_predictions`` (1-D array of mean predicted
        probabilities).
    """

    # sklearn's partial_dependence rejects integer columns.
    # Copy and cast so the caller's DataFrame is not mutated.
    X = X.copy()
    int_cols = X.select_dtypes(include=["int"]).columns
    if len(int_cols) > 0:
        X[int_cols] = X[int_cols].astype("float64")

    feature_index = list(X.columns).index(feature)

    result = partial_dependence(
        model,
        X,
        features=[feature_index],
        kind="average",
        grid_resolution=grid_resolution,
        response_method="predict_proba",
    )

    # result["average"] shape is (1, n_grid_points) for a single
    # feature. With response_method="predict_proba" on a binary
    # classifier, sklearn returns the positive-class probability.
    avg_predictions = result["average"].ravel()

    grid_values = result["grid_values"][0]

    return {
        "values": grid_values,
        "avg_predictions": avg_predictions,
    }


def _get_display_name(feature_name):
    """
    Return the human-readable display name for a feature.
    Falls back to replacing underscores with spaces and
    title-casing if no explicit mapping exists.
    """

    return FEATURE_DISPLAY_NAMES.get(
        feature_name,
        feature_name.replace("_", " ").title(),
    )


def plot_single_pdp(
    grid_values,
    avg_predictions,
    feature_name,
    output_path,
):
    """
    Plot and save a single PDP figure.

    Parameters
    ----------
    grid_values : array-like
        Feature values on the x-axis.
    avg_predictions : array-like
        Average predicted probabilities on the y-axis.
    feature_name : str
        Name of the feature (used in title and labels).
    output_path : str or Path
        Where to save the figure.
    """

    matplotlib.use("Agg")

    display_name = _get_display_name(feature_name)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        grid_values,
        avg_predictions,
        color="#2563eb",
        linewidth=2,
    )

    ax.set_xlabel(display_name, fontsize=12)
    ax.set_ylabel(
        "Predicted P(early readmission)",
        fontsize=12,
    )
    ax.set_title(
        f"Partial Dependence: {display_name}",
        fontsize=14,
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_pdp_plots(
    model=None,
    data_path=None,
    features=None,
    output_dir=None,
    sample_size=500,
    grid_resolution=50,
):
    """
    Generate and save PDP plots for the specified features.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline, optional
        A fitted pipeline. If None, loads from the default path.
    data_path : str or Path, optional
        Path to the cleaned CSV data file. Defaults to
        data/processing/cleaned_diabetic_data.csv.
    features : list of str, optional
        Features to plot. Defaults to DEFAULT_PDP_FEATURES.
    output_dir : str or Path, optional
        Directory to save plots. Defaults to reports/pdp/.
    sample_size : int, optional
        Number of rows to sample from the dataset for computing
        PDP (for performance). Default 500.
    grid_resolution : int, optional
        Number of grid points per feature. Default 50.

    Returns
    -------
    list of dict
        One dict per generated plot with keys:
        ``feature``, ``display_name``, ``output_path``,
        ``grid_min``, ``grid_max``, ``pred_min``, ``pred_max``.
    """

    # Load model if not provided.
    if model is None:
        model = load_model()

    # Load data.
    if data_path is None:
        data_path = (
            REPORTS_DIR.parent
            / "data"
            / "processing"
            / "cleaned_diabetic_data.csv"
        )

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    # Drop target columns to get raw features.
    target_cols = [
        c
        for c in ["readmitted", "early_readmission"]
        if c in df.columns
    ]
    X = df.drop(columns=target_cols)

    # sklearn's partial_dependence requires numerical columns
    # to be floating-point, not integer.
    int_cols = X.select_dtypes(include=["int"]).columns
    X[int_cols] = X[int_cols].astype("float64")

    # Sample for performance.
    if sample_size and len(X) > sample_size:
        X = X.sample(
            n=sample_size,
            random_state=42,
        )

    # Validate features.
    if features is None:
        features = DEFAULT_PDP_FEATURES

    features = validate_features(model, features)

    # Output directory.
    if output_dir is None:
        output_dir = PDP_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots.
    results = []

    for feature in features:

        display_name = _get_display_name(feature)

        print(f"Computing PDP for: {display_name}")

        pdp_result = compute_pdp(
            model,
            X,
            feature,
            grid_resolution=grid_resolution,
        )

        output_path = output_dir / f"pdp_{feature}.png"

        plot_single_pdp(
            grid_values=pdp_result["values"],
            avg_predictions=pdp_result["avg_predictions"],
            feature_name=feature,
            output_path=output_path,
        )

        results.append({
            "feature": feature,
            "display_name": display_name,
            "output_path": output_path,
            "grid_min": float(pdp_result["values"].min()),
            "grid_max": float(pdp_result["values"].max()),
            "pred_min": float(
                pdp_result["avg_predictions"].min()
            ),
            "pred_max": float(
                pdp_result["avg_predictions"].max()
            ),
        })

        print(f"  Saved: {output_path}")

    print(
        f"\nGenerated {len(results)} PDP plot(s) "
        f"in {output_dir}"
    )

    return results


if __name__ == "__main__":
    generate_pdp_plots()
