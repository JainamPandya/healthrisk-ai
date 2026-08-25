from healthrisk.risk import classify_risk


test_probabilities = [
    0.05,
    0.15,
    0.25,
    0.35,
    0.40,
    0.65,
    0.90,
]


for probability in test_probabilities:

    risk = classify_risk(probability)

    print(
        f"Risk score: {probability:.2f} "
        f"→ {risk}"
    )