# ThresholdSelector

ThresholdSelector is for optimizing classification thresholds in binary classification models. It helps you select the optimal threshold for your model based on various performance criteria, with additional statistical confidence metrics.

## Overview

When working with binary classification models, choosing the right threshold is crucial for balancing recall (sensitivity) and precision. This library provides tools to:

1. Find the best threshold that maintains a minimum required recall
2. Calculate confidence intervals for performance metrics
3. Visualize model performance across different thresholds
4. Generate detailed performance reports

## Features

- **Optimal Threshold Selection**: Find the best threshold that meets specified criteria (e.g., minimum recall)
- **Statistical Confidence**: Wilson score intervals for robust confidence bounds
- **Comprehensive Metrics**: Calculate recall, precision, F1-score, specificity, and accuracy
- **Visualization**: Plot performance metrics across different thresholds
- **Reporting**: Generate detailed performance reports and export to pandas DataFrames

## Installation

```
pip install threshold-selector
```

## Usage

### Basic Usage

```python
from threshold_selector import ThresholdSelector, ConfusionMatrixMetrics

# Create sample confusion matrix metrics for different thresholds
# Each entry is (threshold, true_positives, true_negatives, false_positives, false_negatives)
sample_data = [
    ConfusionMatrixMetrics(0.1, 95, 80, 20, 5),    # High recall but more false positives
    ConfusionMatrixMetrics(0.2, 90, 85, 15, 10),   # Good balance
    ConfusionMatrixMetrics(0.3, 85, 90, 10, 15),   # Better precision
    ConfusionMatrixMetrics(0.4, 80, 95, 5, 20),    # Even better precision
    ConfusionMatrixMetrics(0.5, 75, 97, 3, 25),    # Strong precision
]

# Initialize ThresholdSelector
selector = ThresholdSelector(sample_data)

# Find the best threshold that meets recall >= 0.9
best_threshold = selector.find_best_threshold(min_recall=0.9)
print(f"Best threshold: {best_threshold}")

# Visualize metrics
selector.plot_metrics()
```

### Advanced Features

#### Using Confidence Intervals

For a more conservative estimate, use confidence intervals:

```python
# Find best threshold with 95% confidence intervals (more conservative)
best_threshold_ci = selector.find_best_threshold(min_recall=0.9, use_confidence_interval=True)
print(f"Best threshold (with confidence intervals): {best_threshold_ci}")
```

#### Optimizing Different Metrics

Find thresholds that optimize specific metrics:

```python
# Find threshold with optimal F1 score
f1_threshold, f1_value = selector.calculate_optimal_threshold_for_metric('f1')
print(f"Optimal F1 threshold: {f1_threshold} (F1 = {f1_value:.4f})")

# Find threshold with optimal precision
precision_threshold, precision_value = selector.calculate_optimal_threshold_for_metric('precision')
print(f"Optimal precision threshold: {precision_threshold} (Precision = {precision_value:.4f})")
```

#### Exporting Data

Export metrics to a pandas DataFrame for further analysis:

```python
# Export to DataFrame
df = selector.export_to_dataframe()
print(df[['threshold', 'recall', 'precision', 'f1', 'recall_lower_ci', 'recall_upper_ci']])
```

#### Visualization Options

Customize your metric plots:

```python
# Save plot to file with custom figure size
selector.plot_metrics(save_path='threshold_metrics.png', figsize=(12, 8))
```

## API Reference

### ConfusionMatrixMetrics

A data class for storing confusion matrix metrics at a specific threshold.

**Attributes:**
- `threshold`: Classification score threshold (between 0 and 1)
- `true_positives`: Number of correctly predicted positive instances
- `true_negatives`: Number of correctly predicted negative instances
- `false_positives`: Number of incorrectly predicted positive instances (Type I error)
- `false_negatives`: Number of incorrectly predicted negative instances (Type II error)

### ThresholdSelector

The main class for selecting optimal thresholds.

**Constructor:**
```python
ThresholdSelector(metrics_data: List[ConfusionMatrixMetrics])
```

**Main Methods:**

- `find_best_threshold(min_recall=0.9, use_confidence_interval=False)`: Find highest threshold meeting minimum recall
- `calculate_optimal_threshold_for_metric(metric='f1', use_confidence_interval=False)`: Find threshold optimizing a metric
- `plot_metrics(save_path=None, figsize=(10, 6))`: Visualize performance metrics across thresholds
- `get_metrics_report()`: Generate detailed metrics report for all thresholds
- `export_to_dataframe()`: Export metrics to pandas DataFrame

## Code Quality and Testing

The code is designed with quality and robustness in mind:

1. **Comprehensive Input Validation**: Checks for invalid inputs, duplicate thresholds, and other edge cases
2. **Exception Handling**: Proper error messages and exception handling
3. **Detailed Logging**: Configurable logging for debugging and monitoring
4. **Exhaustive Test Suite**: Unit tests covering all functionality and edge cases
5. **Documentation**: Thorough docstrings and comments

### Test Suite

The test suite (`test_threshold_selector.py`) thoroughly validates all functionality:

- Data validation (e.g., empty data, duplicate thresholds, out-of-bounds values)
- Metric calculations and confidence interval computation
- Threshold selection with and without confidence intervals
- Edge cases and zero-division handling
- Visualization and export functionality
- Integration tests for full workflow validation

To run tests:

```
python -m unittest test_threshold_selector.py
```

## Implementation Details

### Performance Metrics

The library calculates the following metrics:

- **Recall/Sensitivity**: TP / (TP + FN)
- **Precision**: TP / (TP + FP)
- **F1 Score**: 2 * (precision * recall) / (precision + recall)
- **Specificity**: TN / (TN + FP)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

### Confidence Intervals

Wilson score intervals are used for calculating confidence bounds, which are more robust than normal approximation, especially for:
- Small sample sizes
- Extreme probabilities (near 0 or 1)

### Data Validation

The library performs thorough validation:
- No duplicate thresholds
- All thresholds in [0,1] range
- No negative values in confusion matrices
- Non-empty input data

