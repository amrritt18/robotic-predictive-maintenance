
import pandas as pd
import streamlit as st

from src.config.settings import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
from src.models.loader import load_artifacts
from src.models.predict import predict_failure

import json
import pickle

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
)

st.title("Machine Failure Detection")

st.write(
    "Estimate machine-failure probability from current machine "
    "operating conditions using a trained machine-learning model."
)

st.caption(
    "This application is a project demonstration, not a validated "
    "industrial safety or maintenance system."
)


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not METADATA_PATH.is_file():
    st.error(f"Metadata file not found: {METADATA_PATH}")
    st.stop()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, metadata = get_artifacts()
    threshold = float(metadata["decision_threshold"])
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()


# --------------------------------------------------
# Input form
# --------------------------------------------------

st.subheader("Machine Operating Conditions")

with st.form("machine_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Product type",
            options=["L", "M", "H"],
            help="Product quality category in the AI4I dataset.",
        )

        air_temp = st.number_input(
            "Air temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1,
        )

        process_temp = st.number_input(
            "Process temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1,
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational speed (rpm)",
            min_value=0,
            max_value=10000,
            value=1500,
            step=10,
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=40.0,
            step=0.5,
        )

        tool_wear = st.number_input(
            "Tool wear (min)",
            min_value=0,
            max_value=1000,
            value=100,
            step=1,
        )

    submitted = st.form_submit_button(
        "Analyze Machine",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:
    if process_temp <= air_temp:
        st.warning(
            "Process temperature is not greater than air temperature. "
            "Check whether these input values are appropriate."
        )

    if rotational_speed <= 0:
        st.error("Rotational speed must be greater than zero.")
        st.stop()

    input_data = pd.DataFrame(
        [
            {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": machine_type,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    try:
        result = predict_failure(
            model=model,
            input_data=input_data,
            threshold=threshold,
        )

        failure_probability = float(
            result["failure_probability"][0]
        )
        predicted_failure = bool(
            result["predicted_failure"][0]
        )

    except (ValueError, TypeError) as exc:
        st.error(f"Invalid prediction input: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Estimated failure probability",
            f"{failure_probability:.1%}",
        )

    with metric_col2:
        st.metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

    st.progress(failure_probability)

    if predicted_failure:
        st.error(
            "Failure alert: the predicted probability meets or "
            "exceeds the configured decision threshold."
        )
    else:
        st.success(
            "No failure alert: the predicted probability is below "
            "the configured decision threshold."
        )

    st.caption(
        "The probability is a model estimate, not a guarantee of "
        "failure or machine health. Verify predictions against real "
        "operating conditions and maintenance procedures."
    )

    with st.expander("View submitted inputs"):
        st.dataframe(input_data, use_container_width=True)

    with st.expander("About this model"):
        st.write(f"Model: {metadata['model_name']}")
        st.write(f"Positive class: {metadata['positive_class']}")
        st.write(f"Threshold: {threshold}")
        st.write(
            "The threshold was selected using training out-of-fold "
            "predictions, not the held-out test set."
        )