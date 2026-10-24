
import pandas as pd

from src.config.settings import FEATURE_COLUMNS


def validate_input(input_data: pd.DataFrame) -> None:
    """Validate the structure and completeness of prediction inputs."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if input_data.empty:
        raise ValueError("Input data must contain at least one row.")

    missing_columns = set(FEATURE_COLUMNS) - set(input_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required features: {sorted(missing_columns)}"
        )

    if input_data[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Input features cannot contain missing values.")
