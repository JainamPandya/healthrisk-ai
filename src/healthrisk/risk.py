def classify_risk(probability: float) -> str:
    """
    Convert model risk score into a simple risk category.
    """

    if probability < 0.20:
        return "Low"

    if probability < 0.40:
        return "Moderate"

    return "High"