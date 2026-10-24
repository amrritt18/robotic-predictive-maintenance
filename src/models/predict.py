
import numpy as np
import pandas as pd

from src.config.settings import FEATURE_COLUMNS
from src.data.validation import validate_input


def predict_failure(
    model,
    input_data: pd.DataFrame,
    threshold: float,
) -> dict:
    """Return failure probabilities and threshold-based predictions."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
    ):
        raise TypeError("Threshold must be numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    validate_input(input_data)

    # Preserve the feature order used by the trained pipeline.
    model_input = input_data[FEATURE_COLUMNS]

    probabilities = model.predict_proba(model_input)[:, 1]

    return {
        "failure_probability": probabilities,
        "predicted_failure": (probabilities >= threshold).astype(int),
        "threshold": float(threshold),
    }
