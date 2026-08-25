"""
Generate Partial Dependence Plots for the HealthRisk AI model.

Loads the trained LightGBM pipeline and cleaned dataset,
then generates PDP plots for the most important numerical
features and saves them to reports/pdp/.

Usage:
    .venv\\Scripts\\python.exe scripts/generate_pdp.py
"""

from pathlib import Path

from healthrisk.config import MODELS_DIR, REPORTS_DIR
from healthrisk.pdp import (
    generate_pdp_plots,
    load_model,
    DEFAULT_PDP_FEATURES,
)


def main():
    model_path = MODELS_DIR / "healthrisk_lightgbm.joblib"
    data_path = (
        REPORTS_DIR.parent
        / "data"
        / "processing"
        / "cleaned_diabetic_data.csv"
    )

    print("=" * 60)
    print("HealthRisk AI — Partial Dependence Plot Generator")
    print("=" * 60)
    print(f"\nModel:    {model_path}")
    print(f"Data:     {data_path}")
    print(f"Features: {DEFAULT_PDP_FEATURES}")
    print()

    model = load_model(model_path)

    results = generate_pdp_plots(
        model=model,
        data_path=data_path,
        features=DEFAULT_PDP_FEATURES,
        sample_size=500,
        grid_resolution=50,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for entry in results:
        print(
            f"  {entry['display_name']:<35s} "
            f"P(readmit) range: "
            f"[{entry['pred_min']:.3f}, "
            f"{entry['pred_max']:.3f}]"
        )

    print(f"\n{len(results)} PDP plots saved to: "
          f"{results[0]['output_path'].parent}")


if __name__ == "__main__":
    main()
