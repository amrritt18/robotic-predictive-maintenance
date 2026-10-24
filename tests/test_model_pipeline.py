from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import METADATA_PATH, MODEL_PATH
from src.models.loader import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_artifacts_exist():
    assert MODEL_PATH.is_file()
    assert METADATA_PATH.is_file()


def test_model_and_metadata_load():
    model, metadata = load_artifacts()

    assert hasattr(model, "predict_proba")
    assert metadata["model_name"] == "Hist_Gradient_Boosting"


def test_decision_threshold_is_fixed():
    _, metadata = load_artifacts()

    assert metadata["decision_threshold"] == 0.25


def test_prediction_probability_is_valid():
    model, _ = load_artifacts()

    sample = pd.DataFrame(
        [{
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100,
            "Type": "L",
        }]
    )

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.isclose(probabilities.sum(axis=1)[0], 1.0)
