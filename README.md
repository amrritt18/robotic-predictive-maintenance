# Robotic Predictive Maintenance Using Machine Learning

A machine-learning project for identifying potential machine failures from current operating conditions. The project covers exploratory data analysis, leakage-aware preprocessing, model comparison, decision-threshold selection, held-out evaluation, automated testing, and a Streamlit demonstration application.

## 1. Project Overview

Unexpected machine failures can interrupt production, increase maintenance costs, and reduce equipment availability. This project explores a machine-learning approach to classifying whether a machine is experiencing a failure based on its current operating conditions.

**Prediction objective:** Binary classification of `Machine failure` at the current observation.

This is not a remaining-useful-life (RUL) estimation project and does not predict a specified number of hours or days into the future.

### Project goals

- Identify machine-failure observations.
- Balance missed failures against false alarms.
- Compare multiple classification models using stratified cross-validation.
- Select a decision threshold using training out-of-fold predictions.
- Evaluate the selected model on a held-out test set.
- Demonstrate predictions through a Streamlit application.

## 2. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset.

The dataset contains 10,000 observations and 14 original columns. The target is `Machine failure`.

The dataset is synthetic and is intended for research and educational experimentation. Results on this dataset do not establish performance on real industrial equipment.

### Input features

The model uses six operating-condition features:

| Feature | Description |
|---|---|
| `Type` | Product quality category |
| `Air temperature [K]` | Air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Rotational speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Accumulated tool wear |

### Excluded columns

- `UDI` and `Product ID` are excluded because they are identifiers.
- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded because they represent failure-mode labels and could leak information about the target.

The project uses `Machine failure` as the authoritative target.

## 3. Workflow

The project follows a staged machine-learning workflow:

1. **Problem definition:** Define the prediction objective, assumptions, success criteria, and limitations.
2. **Exploratory data analysis:** Examine class distribution, data quality, feature distributions, outliers, and correlations.
3. **Preprocessing:** Split data into training and test sets, handle numerical and categorical features, and prevent preprocessing leakage during cross-validation.
4. **Baseline modeling:** Compare a dummy baseline and multiple classification algorithms.
5. **Model experiments:** Generate training out-of-fold probabilities and compare candidate models and decision thresholds.
6. **Final evaluation:** Evaluate the selected model on the held-out test set.
7. **Threshold optimization:** Document and verify the threshold-selection decision.
8. **Final model analysis:** Summarize results, limitations, and saved artifacts.
9. **Deployment and packaging:** Serve predictions through Streamlit, test the saved model, and document project setup.

## 4. Model Selection

The evaluated candidates included:

- Dummy classifier
- Logistic regression
- Decision tree
- Random forest
- Extra trees
- Histogram-based gradient boosting

The selected model is **HistGradientBoostingClassifier**, configured with class balancing.

Model selection considered average precision, recall, precision, F1-score, balanced accuracy, and ROC-AUC. Average precision is particularly informative when the positive class is rare.

The selected model pipeline includes preprocessing so that inference uses the same transformations as training.

## 5. Decision Threshold

The selected decision threshold is **0.25**.

A failure alert is raised when:

`Predicted failure probability >= 0.25`

The threshold was selected from training out-of-fold predictions by maximizing precision among the tested thresholds satisfying the project criterion of recall greater than or equal to 0.85.

This criterion is a project design choice, not an industry-validated operating requirement. The held-out test set was not used to select the threshold.

## 6. Held-Out Test Results

The selected pipeline was evaluated on a held-out test set of 2,000 observations.

| Metric | Result |
|---|---:|
| Accuracy | 95.70% |
| Precision | 43.38% |
| Recall | 86.76% |
| F1-score | 57.84% |
| Average precision | 82.46% |
| ROC-AUC | 97.44% |
| Balanced accuracy | 91.39% |

### Confusion matrix counts

| Outcome | Count |
|---|---:|
| True positives | 59 |
| False positives | 77 |
| True negatives | 1,855 |
| False negatives | 9 |

At the selected threshold, the model detected 59 of the 68 failure observations in the test set and missed 9. It also generated 77 false alarms.

The recall is relatively high, but precision is lower because the dataset has a low failure prevalence. A failure alert should therefore be treated as a signal for further investigation, not proof that a failure has occurred.

These figures are estimates from one held-out split. Performance may differ on new data.

## 7. Streamlit Application

The application accepts the six operating-condition features and displays:

- Estimated failure probability
- Configured decision threshold
- Failure alert or no-alert result
- Submitted input values
- Model information

### Run locally

Ensure that the saved model and metadata are available at:

- `artifacts/models/final_model_pipeline.joblib`
- `artifacts/reports/final_model_metadata.json`

Then run:

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

The application loads the saved model and does not retrain it when a user submits inputs.

## 8. Testing

Automated tests are located in the `tests/` directory.

- `test_model_pipeline.py` checks artifact availability, model loading, the configured threshold, and valid model probabilities.
- `test_prediction.py` checks input validation, missing features, missing values, threshold validation, prediction outputs, and feature-order consistency.

Run the complete test suite from the project root:

```bash
uv run pytest tests/ -v
```

**Current status:** All 10 tests pass.

The tests validate the saved model artifacts and application prediction logic. They do not establish industrial reliability or guarantee performance on unseen real-world equipment.

## 9. Project Structure

```text
robotic-predictive-maintenance/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   ├── reports/
│   └── processed/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_problem_definition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_experiments.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_threshold_optimization.ipynb
│   └── 08_final_model_analysis.ipynb
├── src/
├── tests/
│   └── test_model_pipeline.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

The tree shows the intended project layout; optional generated artifacts may not be present in a fresh checkout.

## 10. Reproducibility

The project uses Python 3.12 and `uv` for dependency management.

- `pyproject.toml` declares the project dependencies.
- `uv.lock` records the resolved dependency versions.
- `requirements.txt` provides an exported requirements list.

To synchronize the environment:

```bash
uv sync
```

To check the lockfile:

```bash
uv lock --check
```

## 11. Limitations and Future Improvements

- The dataset is synthetic, so real-world generalization is unknown.
- False alarms may lead to unnecessary inspections or maintenance.
- Some failure observations are missed even at the selected threshold.
- The selected threshold reflects a project-level recall criterion rather than a measured industrial cost function.
- The test results come from one stratified train/test split.
- Input ranges in the demonstration app are basic validation safeguards, not certified operating limits.
- The application has not been validated as a real-time monitoring or industrial safety system.

Potential future improvements include external validation on real machine telemetry, calibration analysis, cost-based threshold selection, monitoring for data drift, and integration with a properly validated maintenance workflow.

## 12. Conclusion

This project demonstrates an end-to-end predictive-maintenance classification workflow, from data exploration and model selection to threshold optimization, held-out evaluation, testing, and a working Streamlit interface.

The results show promising classification performance on the AI4I 2020 dataset, while also highlighting the practical trade-off between detecting failures and generating false alarms. Real-world deployment would require further validation and operational testing.
