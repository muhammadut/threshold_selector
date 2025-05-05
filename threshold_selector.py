from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

# Setup logging with appropriate level and format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrixMetrics:
    """
    Data class for storing confusion matrix metrics at a specific threshold.
    
    Attributes:
        threshold: Classification confidence score threshold (between 0 and 1)
        true_positives: Number of correctly predicted positive instances
        true_negatives: Number of correctly predicted negative instances
        false_positives: Number of incorrectly predicted positive instances (Type I error)
        false_negatives: Number of incorrectly predicted negative instances (Type II error)
    """
    threshold: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class ThresholdSelector:
    """
    A class for selecting the optimal threshold for a binary classification model
    that meets specified performance criteria, with statistical confidence metrics.
    
    This class provides functionality to:
    1. Find the best threshold that yields recall >= min_recall
    2. Calculate confidence intervals for recall
    3. Generate visualizations of model performance metrics across thresholds
    4. Produce detailed performance reports
    """
    
    def __init__(self, metrics_data: List[ConfusionMatrixMetrics]):
        """
        Initialize the ThresholdSelector with confusion matrix metrics.
        
        Args:
            metrics_data: List of ConfusionMatrixMetrics objects containing 
                         confusion matrix data for different thresholds
        
        Raises:
            ValueError: If input data validation fails
        """
        # Sort metrics by threshold for consistent processing
        self.metrics_data = sorted(metrics_data, key=lambda x: x.threshold)
        # Validate input data before proceeding
        self._validate_data()
        # Dictionary to store computed metrics for each threshold
        self.threshold_metrics = {}
        # Compute all metrics during initialization for efficiency
        self._compute_all_metrics()
        
        logger.info(f"Initialized ThresholdSelector with {len(metrics_data)} thresholds")

    def _validate_data(self):
        """
        Validate the input metrics data.
        
        Checks:
        1. Non-empty input
        2. No duplicate thresholds
        3. Thresholds are within valid range [0,1]
        4. No negative values in confusion matrices
        
        Raises:
            ValueError: If any validation check fails
        """
        # Check if we have any data points
        if not self.metrics_data:
            logger.error("No metrics data provided")
            raise ValueError("No metrics data provided.")
        
        # Check for duplicate thresholds and valid values
        seen = set()
        for m in self.metrics_data:
            # Check for duplicate thresholds
            if m.threshold in seen:
                logger.error(f"Duplicate threshold found: {m.threshold}")
                raise ValueError(f"Duplicate threshold: {m.threshold}")
            seen.add(m.threshold)
            
            # Check threshold bounds
            if not (0 <= m.threshold <= 1):
                logger.error(f"Threshold {m.threshold} outside valid range [0,1]")
                raise ValueError(f"Threshold {m.threshold} out of bounds [0,1]")
            
            # Check for negative counts in confusion matrix
            if any(x < 0 for x in [m.true_positives, m.true_negatives, m.false_positives, m.false_negatives]):
                logger.error(f"Negative values found in confusion matrix for threshold {m.threshold}")
                raise ValueError(f"Negative values in confusion matrix for threshold {m.threshold}")
        
        logger.debug("Data validation passed")

    def _compute_all_metrics(self):
        """
        Compute and store all metrics for each threshold.
        
        For each threshold, calculates:
        1. Recall, precision, F1 score, and specificity
        2. Confidence intervals for recall
        
        This pre-computation improves performance for subsequent operations.
        """
        logger.debug("Computing metrics for all thresholds")
        
        for metric in self.metrics_data:
            # Calculate classification metrics
            recall, precision, f1, specificity = self._calculate_metrics(
                metric.true_positives, metric.true_negatives,
                metric.false_positives, metric.false_negatives
            )
            
            # Calculate confidence intervals for recall
            lower_ci, upper_ci = self._calculate_recall_confidence_interval(
                metric.true_positives, metric.false_negatives
            )
            
            # Store all metrics in dictionary for easy access
            self.threshold_metrics[metric.threshold] = {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "specificity": specificity,
                "recall_lower_ci": lower_ci,
                "recall_upper_ci": upper_ci,
                # Store counts for potential further analysis
                "tp": metric.true_positives,
                "tn": metric.true_negatives,
                "fp": metric.false_positives,
                "fn": metric.false_negatives,
                # Additional metrics that might be useful
                "accuracy": (metric.true_positives + metric.true_negatives) / 
                           (metric.true_positives + metric.true_negatives + 
                            metric.false_positives + metric.false_negatives)
                if (metric.true_positives + metric.true_negatives + 
                    metric.false_positives + metric.false_negatives) > 0 else 0.0
            }
        
        logger.debug(f"Computed metrics for {len(self.threshold_metrics)} thresholds")

    @staticmethod
    def _calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
        """
        Calculate classification metrics from confusion matrix counts.
        
        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives
            
        Returns:
            Tuple of (recall, precision, f1, specificity)
            
        Note:
            Handles division by zero cases gracefully
        """
        # Calculate recall (sensitivity, true positive rate)
        # Formula: TP / (TP + FN)
        recall = tp / (tp + fn) if tp + fn else 0.0
        
        # Calculate precision (positive predictive value)
        # Formula: TP / (TP + FP)
        precision = tp / (tp + fp) if tp + fp else 0.0
        
        # Calculate F1 score (harmonic mean of precision and recall)
        # Formula: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        
        # Calculate specificity (true negative rate)
        # Formula: TN / (TN + FP)
        specificity = tn / (tn + fp) if tn + fp else 0.0
        
        return recall, precision, f1, specificity

    @staticmethod
    def _calculate_recall_confidence_interval(tp: int, fn: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for recall using Wilson score interval method.
        
        The Wilson score interval is more accurate than normal approximation,
        especially for small sample sizes and extreme probabilities.
        
        Args:
            tp: True positives
            fn: False negatives
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        total = tp + fn
        
        # Handle edge case of no positive instances
        if total == 0:
            return 0.0, 0.0
        
        # Wilson score interval is preferred over normal approximation
        # especially for small samples or extreme probabilities
        return proportion_confint(tp, total, alpha=1 - confidence, method="wilson")

    def find_best_threshold(self, min_recall: float = 0.9, use_confidence_interval: bool = False) -> Optional[float]:
        """
        Find the best (highest) threshold that meets the minimum recall requirement.
        
        A higher threshold typically reduces false positives at the expense of
        potentially increasing false negatives. This method finds the highest
        threshold that still maintains the required recall level.
        
        Args:
            min_recall: Minimum acceptable recall value (default: 0.9)
            use_confidence_interval: If True, use the lower bound of the CI
                                   for a more conservative estimate
                                   
        Returns:
            The best threshold meeting criteria, or None if no threshold qualifies
            
        Raises:
            ValueError: If min_recall is not in [0, 1]
        """
        # Validate input
        if not (0 <= min_recall <= 1):
            logger.error(f"Invalid min_recall value: {min_recall}")
            raise ValueError("min_recall must be in [0, 1]")
        
        if not self.threshold_metrics:
            logger.warning("No metrics to evaluate.")
            return None
        
        logger.info(f"Finding best threshold with min_recall={min_recall}, "
                   f"use_confidence_interval={use_confidence_interval}")
        
        # Collect all thresholds that meet our criteria
        valid_thresholds = []
        for t, metrics in self.threshold_metrics.items():
            # Use confidence interval lower bound if requested (more conservative)
            recall_value = metrics['recall_lower_ci'] if use_confidence_interval else metrics['recall']
            
            if recall_value >= min_recall:
                valid_thresholds.append(t)
        
        # Check if we found any valid thresholds
        if not valid_thresholds:
            message = f"No thresholds found with recall ≥ {min_recall}"
            if use_confidence_interval:
                # Provide a helpful explanation about confidence intervals
                message += (
                    " when using confidence intervals. This is because confidence "
                    "intervals provide a more conservative estimate. The lower bounds "
                    "of the confidence intervals for recall are below the minimum "
                    "required value. Consider:\n"
                    "1. Using a lower minimum recall requirement\n"
                    "2. Using point estimates instead of confidence intervals\n"
                    "3. Increasing your sample size to narrow confidence intervals\n"
                    "Run export_to_dataframe() to examine the actual recall values and "
                    "their confidence intervals."
                )
            logger.warning(message)
            print(message)  # Print directly to user for visibility
            return None
        
        # Return the highest threshold that meets our criteria
        best_threshold = max(valid_thresholds)
        logger.info(f"Selected best threshold: {best_threshold}")
        return best_threshold

    def plot_metrics(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot classification metrics across different thresholds.
        
        Creates a visualization showing how recall, precision, F1 score change
        with different threshold values. Also plots confidence intervals for recall.
        
        Args:
            save_path: If provided, save the plot to this path instead of displaying
            figsize: Figure dimensions (width, height) in inches
            
        Returns:
            None
        """
        # Check if we have enough data points to create a meaningful plot
        if len(self.threshold_metrics) < 2:
            logger.warning("Insufficient data points to plot metrics.")
            return
        
        logger.debug("Creating metrics plot")
        
        # Extract data for plotting
        thresholds = sorted(self.threshold_metrics.keys())
        recalls = [self.threshold_metrics[t]["recall"] for t in thresholds]
        precisions = [self.threshold_metrics[t]["precision"] for t in thresholds]
        f1s = [self.threshold_metrics[t]["f1"] for t in thresholds]
        lower_cis = [self.threshold_metrics[t]["recall_lower_ci"] for t in thresholds]
        upper_cis = [self.threshold_metrics[t]["recall_upper_ci"] for t in thresholds]
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot metrics
        plt.plot(thresholds, recalls, 'o-', label="Recall", linewidth=2)
        plt.plot(thresholds, precisions, 's-', label="Precision", linewidth=2)
        plt.plot(thresholds, f1s, '^-', label="F1 Score", linewidth=2)
        
        # Plot confidence intervals for recall
        plt.fill_between(thresholds, lower_cis, upper_cis, 
                        alpha=0.2, color='blue', 
                        label="95% CI for Recall")
        
        # Add a reference line for the common minimum recall requirement of 0.9
        plt.axhline(y=0.9, color='red', linestyle='--', 
                   label="Recall = 0.9", linewidth=1.5)
        
        # Formatting
        plt.title("Classification Metrics vs. Threshold", fontsize=14)
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        
        # Add annotations for the best threshold
        best_threshold = self.find_best_threshold(min_recall=0.9)
        if best_threshold is not None:
            best_metrics = self.threshold_metrics[best_threshold]
            plt.axvline(x=best_threshold, color='green', linestyle=':', 
                      label=f"Best Threshold = {best_threshold}", linewidth=1.5)
            plt.annotate(f"Best Threshold: {best_threshold}",
                       xy=(best_threshold, best_metrics['recall']),
                       xytext=(best_threshold+0.05, best_metrics['recall']+0.05),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                       fontsize=10)
        
        # Set axis limits
        plt.ylim(0, 1.05)
        plt.xlim(min(thresholds)-0.05, max(thresholds)+0.05)
        
        # Add grid for readability
        plt.grid(True, alpha=0.3)
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            logger.debug("Plot displayed")

    def get_metrics_report(self) -> Dict:
        """
        Generate a comprehensive report of metrics for all thresholds.
        
        Returns:
            Dictionary with threshold values as keys and metric dictionaries as values
        """
        logger.debug("Generating metrics report")
        return self.threshold_metrics
    
    def calculate_optimal_threshold_for_metric(
        self, 
        metric: str = 'f1', 
        use_confidence_interval: bool = False
    ) -> Tuple[float, float]:
        """
        Find the threshold that optimizes a specified metric.
        
        Args:
            metric: The metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            use_confidence_interval: If True and metric is 'recall', use lower bound
            
        Returns:
            Tuple of (optimal_threshold, metric_value)
            
        Raises:
            ValueError: If the specified metric is not supported
        """
        supported_metrics = ['f1', 'precision', 'recall', 'accuracy']
        if metric not in supported_metrics:
            raise ValueError(f"Metric must be one of {supported_metrics}")
        
        # For recall with confidence intervals, we'll use the lower bound
        if metric == 'recall' and use_confidence_interval:
            metric_key = 'recall_lower_ci'
        else:
            metric_key = metric
        
        best_value = -1
        best_threshold = None
        
        for threshold, metrics in self.threshold_metrics.items():
            current_value = metrics[metric_key]
            if current_value > best_value:
                best_value = current_value
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold} with value {best_value:.4f}")
        return best_threshold, best_value
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export all metrics to a pandas DataFrame for further analysis.
        
        Returns:
            DataFrame with all metrics and thresholds
        """
        # Create a list of records
        records = []
        for threshold, metrics in self.threshold_metrics.items():
            record = {'threshold': threshold}
            record.update(metrics)
            records.append(record)
        
        # Convert to DataFrame and sort by threshold
        df = pd.DataFrame(records)
        df = df.sort_values('threshold').reset_index(drop=True)
        
        logger.debug(f"Exported metrics to DataFrame with {len(df)} rows")
        return df

    def explain_ci_behavior(self, min_recall: float = 0.9) -> None:
        """
        Provides an explanation of how confidence intervals affect threshold selection.
        
        This method helps users understand the difference between point estimates
        and confidence intervals, especially when no thresholds meet the criteria
        with confidence intervals enabled.
        
        Args:
            min_recall: The minimum recall requirement to explain
            
        Returns:
            None (prints explanation to console)
        """
        # Get standard thresholds and CI thresholds
        standard_threshold = self.find_best_threshold(min_recall, use_confidence_interval=False)
        ci_threshold = self.find_best_threshold(min_recall, use_confidence_interval=True)
        
        # Create a small summary table of key thresholds
        df = self.export_to_dataframe()
        key_columns = ['threshold', 'recall', 'recall_lower_ci', 'recall_upper_ci']
        
        print("\n=== Confidence Interval Explanation ===\n")
        print(f"Using point estimates, the best threshold is: {standard_threshold}")
        print(f"Using confidence intervals, the best threshold is: {ci_threshold}")
        print("\nWhy the difference? Confidence intervals provide a more conservative estimate.")
        print("When using confidence intervals, we require the LOWER BOUND of the recall")
        print(f"confidence interval to be ≥ {min_recall}, not just the point estimate.")
        print("\nHere's a summary of your data showing the confidence intervals:")
        print(df[key_columns].to_string(index=False, float_format="%.4f"))
        
        print("\nOptions if you're getting None with confidence intervals:")
        print("1. Use a lower minimum recall requirement")
        print("2. Use point estimates instead (set use_confidence_interval=False)")
        print("3. Increase your sample size to narrow confidence intervals")
        print("4. Examine the DataFrame above to choose a threshold that balances")
        print("   your tolerance for risk with your recall requirements")


# Example usage function
def find_best_threshold(
    confusion_matrix_data: List[Tuple[float, int, int, int, int]],
    min_recall: float = 0.9,
    use_confidence_interval: bool = False
) -> Optional[float]:
    """
    Find the best threshold that yields a recall >= min_recall.
    
    This is the main function requested in the assignment.
    
    Args:
        confusion_matrix_data: List of tuples (threshold, TP, TN, FP, FN)
        min_recall: Minimum acceptable recall value
        use_confidence_interval: Whether to use confidence intervals
        
    Returns:
        The best threshold, or None if no threshold meets criteria
        
    Example:
        >>> data = [(0.1, 95, 80, 20, 5), (0.2, 90, 85, 15, 10), ...]
        >>> find_best_threshold(data, min_recall=0.9)
        0.2
    """
    # Convert input data to ConfusionMatrixMetrics objects
    metrics_data = [
        ConfusionMatrixMetrics(t, tp, tn, fp, fn)
        for t, tp, tn, fp, fn in confusion_matrix_data
    ]
    
    # Create ThresholdSelector and find best threshold
    selector = ThresholdSelector(metrics_data)
    result = selector.find_best_threshold(min_recall, use_confidence_interval)
    
    # If using CI and no threshold found, provide additional explanation
    if result is None and use_confidence_interval:
        print("\nFor more details on confidence intervals and threshold selection:")
        print("Call selector.explain_ci_behavior() for a detailed explanation")
    
    return result


# Example demonstration code
def example_usage():
    """Demonstrate the usage of ThresholdSelector with realistic data."""
    # Sample confusion matrix metrics for different thresholds
    sample_data = [
        ConfusionMatrixMetrics(0.1, 95, 80, 20, 5),    # High recall but more FP
        ConfusionMatrixMetrics(0.2, 90, 85, 15, 10),   # Good balance
        ConfusionMatrixMetrics(0.3, 85, 90, 10, 15),   # Better precision
        ConfusionMatrixMetrics(0.4, 80, 95, 5, 20),    # Even better precision
        ConfusionMatrixMetrics(0.5, 75, 97, 3, 25),    # Strong precision
        ConfusionMatrixMetrics(0.6, 70, 98, 2, 30),    # Very high precision
        ConfusionMatrixMetrics(0.7, 65, 99, 1, 35),    # Extremely high precision
        ConfusionMatrixMetrics(0.8, 60, 99, 1, 40),    # Extreme precision, low recall
        ConfusionMatrixMetrics(0.9, 55, 100, 0, 45),   # Perfect precision, lowest recall
    ]
    
    # Initialize ThresholdSelector
    selector = ThresholdSelector(sample_data)
    
    # Find the best threshold that meets recall >= 0.9
    best_threshold = selector.find_best_threshold(min_recall=0.9)
    print(f"Best threshold: {best_threshold}")
    
    # Find best threshold with confidence intervals (more conservative)
    best_threshold_ci = selector.find_best_threshold(min_recall=0.9, use_confidence_interval=True)
    print(f"Best threshold (with confidence intervals): {best_threshold_ci}")
    
    # If CI approach returns None, explain why
    if best_threshold_ci is None and best_threshold is not None:
        selector.explain_ci_behavior(min_recall=0.9)
    
    # Plot metrics
    selector.plot_metrics()
    
    # Export to DataFrame for further analysis
    df = selector.export_to_dataframe()
    print("\nMetrics DataFrame:")
    print(df[['threshold', 'recall', 'precision', 'f1', 'recall_lower_ci', 'recall_upper_ci']].head())
    
    # Find optimal thresholds for different metrics
    f1_threshold, f1_value = selector.calculate_optimal_threshold_for_metric('f1')
    print(f"\nOptimal threshold for F1 score: {f1_threshold} (F1 = {f1_value:.4f})")
    
    precision_threshold, precision_value = selector.calculate_optimal_threshold_for_metric('precision')
    print(f"Optimal threshold for precision: {precision_threshold} (Precision = {precision_value:.4f})")


if __name__ == "__main__":
    example_usage()
