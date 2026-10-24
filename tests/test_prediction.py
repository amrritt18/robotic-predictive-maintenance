
import numpy as np
import pandas as pd
import pytest

from src.config.settings import FEATURE_COLUMNS
from src.data.validation import validate_input
from src.models.loader import load_artifacts
from src.models.predict import predict_failure


@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
        "Type": "L",
    }])


@pytest.fixture
def artifacts():
    return load_artifacts()


def test_valid_input_passes_validation(sample_input):
    validate_input(sample_input)


def test_missing_feature_is_rejected(sample_input):
    sample_input = sample_input.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Missing required features"):
        validate_input(sample_input)


def test_missing_value_is_rejected(sample_input):
    sample_input.loc[0, "Torque [Nm]"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_input(sample_input)


def test_prediction_returns_valid_results(sample_input, artifacts):
    model, metadata = artifacts

    result = predict_failure(
        model,
        sample_input,
        metadata["decision_threshold"],
    )

    assert len(result["failure_probability"]) == 1
    assert 0 <= result["failure_probability"][0] <= 1
    assert result["predicted_failure"][0] in (0, 1)
    assert result["threshold"] == 0.25


def test_invalid_threshold_is_rejected(sample_input, artifacts):
    model, _ = artifacts

    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_failure(model, sample_input, 1.5)


def test_feature_order_does_not_change_prediction(sample_input, artifacts):
    model, metadata = artifacts

    expected = predict_failure(
        model, sample_input, metadata["decision_threshold"]
    )

    shuffled = sample_input[
        list(reversed(FEATURE_COLUMNS))
    ]

    actual = predict_failure(
        model, shuffled, metadata["decision_threshold"]
    )

    np.testing.assert_allclose(
        expected["failure_probability"],
        actual["failure_probability"],
    )
