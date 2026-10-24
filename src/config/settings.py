
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Saved model artifacts
MODEL_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "final_model_pipeline.joblib"
)

METADATA_PATH = (
    ARTIFACTS_DIR
    / "reports"
    / "final_model_metadata.json"
)

# Target and decision configuration
TARGET_COLUMN = "Machine failure"
DEFAULT_DECISION_THRESHOLD = 0.25

# Expected model input schema.
# Keep the order consistent with the trained pipeline.
FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]
