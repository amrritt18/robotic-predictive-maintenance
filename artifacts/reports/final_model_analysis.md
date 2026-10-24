# Final Model Analysis

## Selected Configuration

- Model: Hist_Gradient_Boosting
- Classification threshold: 0.25
- Imbalance strategy: class weighting
- SMOTE: not used
- Random oversampling: not used
- Random undersampling: not used

## Test-Set Results

| Metric | Value |
|---|---:|
| Accuracy | 0.9570 |
| Average Precision | 0.8246 |
| ROC-AUC | 0.9744 |
| Precision | 0.4338 |
| Recall | 0.8676 |
| F1-score | 0.5784 |
| Balanced Accuracy | 0.9139 |

## Confusion Matrix Counts

- True positives: 59
- False positives: 77
- True negatives: 1855
- False negatives: 9

## Interpretation

The model detected 59 of the actual failures
in the held-out test set and missed 9.
It generated 77 false-positive alerts.

The model demonstrates promising failure-ranking performance, but the
precision indicates that false alarms remain an important concern.

## Limitations

1. The dataset is synthetic and may not represent all real industrial machines.
2. The task detects failure at the current observation; it does not predict remaining useful life or future failure lead time.
3. The selected threshold reflects a provisional recall constraint, not a validated industrial cost function.
4. The test set contains a limited number of failures, so estimates have sampling uncertainty.
5. Real deployment requires validation on operational data, monitoring, and periodic model evaluation.
6. Feature importance and operational explanations should be investigated before relying on predictions for maintenance decisions.

## Conclusion

Hist Gradient Boosting was selected based on out-of-fold average precision
and threshold experiments. A threshold of 0.25 was chosen using training
out-of-fold predictions under a minimum recall criterion of 85%.

The held-out test results are reported separately and were not used to
retune the threshold.