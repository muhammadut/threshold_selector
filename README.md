# ThresholdSelector

ThresholdSelector is a Python tool for optimizing classification thresholds in binary classification models. It helps you select the optimal threshold for your model based on various performance criteria, with additional statistical confidence metrics.

## Overview

In production machine learning systems, choosing the right classification threshold is critical for balancing false positives and false negatives based on business constraints. ThresholdSelector offers a statistically sound approach to this optimization problem by:

1. Finding the highest threshold that maintains minimum recall requirements
2. Incorporating confidence intervals for statistically robust decisions
3. Providing comprehensive metrics visualization
4. Supporting threshold optimization for various performance metrics

## Key Features

- **Statistically Rigorous**: Wilson score confidence intervals for robust threshold selection
- **Multi-Metric Support**: Optimize for recall, precision, F1-score, or accuracy 
- **Basic Visualization**: Plot performance metrics across thresholds with confidence bands
- **Validation**: Thorough error handling and input validation
- **Basic Reporting**: Generate performance reports and export to pandas DataFrames
- **Integration-Ready**: Designed for integration into ML pipelines with logging

## Usage

### Basic Usage

```python
from threshold_selector import ThresholdSelector, ConfusionMatrixMetrics

# Create confusion matrix metrics for different thresholds
sample_data = [
    ConfusionMatrixMetrics(0.1, 95, 80, 20, 5),    # High recall, moderate precision
    ConfusionMatrixMetrics(0.3, 85, 90, 10, 15),   # Balanced recall/precision
    ConfusionMatrixMetrics(0.5, 75, 97, 3, 25),    # High precision, lower recall
]

# Initialize selector
selector = ThresholdSelector(sample_data)

# Find threshold with minimum recall of 0.9
best_threshold = selector.find_best_threshold(min_recall=0.9)
print(f"Best threshold: {best_threshold}")

# Visualize metrics
selector.plot_metrics()
```

### Understanding Results

When `find_best_threshold()` returns `None`:

```python
# Using confidence intervals for more conservative estimates
best_threshold_ci = selector.find_best_threshold(min_recall=0.9, use_confidence_interval=True)
# May return None if no threshold's lower CI meets the minimum recall

# For detailed explanation
selector.explain_ci_behavior(min_recall=0.9)
```

## Advanced Features

### Confidence Interval Analysis

For statistically robust threshold selection:

```python
# Find threshold meeting recall ≥ 0.9 with 95% confidence
threshold_ci = selector.find_best_threshold(min_recall=0.9, use_confidence_interval=True)

# Examine confidence intervals
df = selector.export_to_dataframe()
print(df[['threshold', 'recall', 'recall_lower_ci', 'recall_upper_ci']])
```

### Optimizing Different Metrics

Find thresholds that maximize specific metrics:

```python
# Find threshold with optimal F1 score
f1_threshold, f1_value = selector.calculate_optimal_threshold_for_metric('f1')

# Find threshold with optimal precision
prec_threshold, prec_value = selector.calculate_optimal_threshold_for_metric('precision')
```

### Performance Visualization

Generate visual representations of threshold performance:

```python
# Create detailed performance plot
selector.plot_metrics(figsize=(12, 8))

# Save visualization to file
selector.plot_metrics(save_path='threshold_analysis.png')
```

## API Reference

### ConfusionMatrixMetrics

Data class for storing model performance at a specific threshold:

```python
ConfusionMatrixMetrics(
    threshold: float,       # Classification threshold (0-1)
    true_positives: int,    # Correctly predicted positives
    true_negatives: int,    # Correctly predicted negatives
    false_positives: int,   # Type I errors
    false_negatives: int    # Type II errors
)
```

### ThresholdSelector

Core class for threshold optimization analysis:

#### Constructor
```python
ThresholdSelector(metrics_data: List[ConfusionMatrixMetrics])
```

#### Key Methods
- `find_best_threshold(min_recall=0.9, use_confidence_interval=False)`: 
  Find highest threshold meeting minimum recall requirements
  
- `calculate_optimal_threshold_for_metric(metric='f1', use_confidence_interval=False)`: 
  Find threshold that maximizes a specific metric
  
- `plot_metrics(save_path=None, figsize=(10, 6))`: 
  Generate visualization of threshold performance
  
- `get_metrics_report()`: 
  Get comprehensive metrics for all thresholds
  
- `export_to_dataframe()`: 
  Export metrics to pandas DataFrame
  
- `explain_ci_behavior(min_recall=0.9)`:
  Explain confidence interval effects on threshold selection

## Test Suite

The package includes comprehensive tests (`test_threshold_selector.py`) that verify:

- **Input Validation**: Checks for invalid inputs and edge cases
- **Metric Calculation**: Validates correctness of all performance metrics
- **Confidence Intervals**: Tests Wilson score interval calculations
- **Threshold Selection**: Verifies optimal threshold identification
- **Edge Case Handling**: Tests zero counts, extreme values, and error cases
- **Visualization**: Ensures plot generation functions correctly

Run tests with:
```
python -m unittest test_threshold_selector.py
```

## Implementation Details

### Performance Metrics

Key performance metrics calculated:

- **Recall (Sensitivity)**: `TP / (TP + FN)` - Proportion of actual positives correctly identified
- **Precision**: `TP / (TP + FP)` - Proportion of positive predictions that are correct
- **F1 Score**: `2 * (precision * recall) / (precision + recall)` - Harmonic mean of precision and recall
- **Specificity**: `TN / (TN + FP)` - Proportion of actual negatives correctly identified
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)` - Overall correctness

### Confidence Intervals

The tool implements Wilson score intervals for calculating confidence bounds, which are more appropriate than normal approximation for:

- Small sample sizes
- Extreme probabilities (values near 0 or 1)

#### Interpreting and Using Confidence Intervals

Confidence intervals represent the range where the true metric is likely to lie with a specified confidence level (default 95%). When `use_confidence_interval=True`, the tool uses the lower bound of the confidence interval instead of the point estimate, providing a more conservative assessment.

**Practical Interpretation:**

A 95% confidence interval of [0.85, 0.95] for recall means:
- We can be 95% confident that the true recall is at least 0.85
- In production, the recall is highly unlikely to fall below 0.85


If `find_best_threshold()` returns `None` with confidence intervals enabled:

1. Your dataset may be too small, resulting in wide confidence intervals
2. Try using a lower minimum recall requirement
3. Consider collecting more data to narrow confidence intervals
4. Use `explain_ci_behavior()` to visualize the gap between point estimates and confidence bounds

### Data Validation

The tool performs thorough validation:
- Ensures thresholds are in range [0,1]
- Checks for duplicate thresholds
- Validates that confusion matrix counts are non-negative
- Handles edge cases like division by zero







