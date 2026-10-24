
import json
from pathlib import Path
from typing import Any

import joblib

from src.config.settings import METADATA_PATH, MODEL_PATH


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the serialized model pipeline."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(model_path)


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    """Load and validate the model metadata JSON."""
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata artifact not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "decision_threshold" not in metadata:
        raise ValueError(
            "Model metadata is missing 'decision_threshold'."
        )

    threshold = metadata["decision_threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Decision threshold must be a number between 0 and 1."
        )

    return metadata


def load_artifacts(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Any, dict]:
    """Load the model pipeline and its metadata."""
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)

    return model, metadata
