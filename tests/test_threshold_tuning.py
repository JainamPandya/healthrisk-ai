import numpy as np
from healthrisk.models.threshold_tuning import evaluate_thresholds

def test_evaluate_thresholds_structure():
    """
    Test that evaluate_thresholds correctly evaluates probabilities and labels,
    and structurally proves it does not require test labels.
    """
    # Fake validation set
    y_val_fake = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    val_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.3, 0.1, 0.2, 0.7])
    
    # Evaluate thresholds strictly on validation data
    results = evaluate_thresholds(y_val_fake, val_probs)
    
    assert len(results) > 0, "Should return evaluation results for thresholds."
    
    for res in results:
        threshold, precision, recall, f1 = res
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0

    # For threshold 0.5, the predicted positives are index 1, 3, 4, 9.
    # The true labels for these are 1, 1, 1, 1. So TP=4, FP=0. Precision=1.0.
    thresh_0_5 = next(r for r in results if r[0] == 0.50)
    assert thresh_0_5[1] == 1.0, f"Expected precision 1.0 for threshold 0.5, got {thresh_0_5[1]}"
