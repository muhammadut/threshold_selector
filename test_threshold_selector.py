import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import logging
from io import StringIO

# Import the enhanced threshold selector
from threshold_selector import ThresholdSelector, ConfusionMatrixMetrics


class TestThresholdSelector(unittest.TestCase):
    """Comprehensive test suite for the ThresholdSelector class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Sample data for testing
        self.sample_data = [
            ConfusionMatrixMetrics(0.1, 95, 80, 20, 5),
            ConfusionMatrixMetrics(0.2, 90, 85, 15, 10),
            ConfusionMatrixMetrics(0.3, 85, 90, 10, 15),
            ConfusionMatrixMetrics(0.4, 80, 95, 5, 20),
            ConfusionMatrixMetrics(0.5, 75, 97, 3, 25),
        ]

        # Configure log capture - UPDATED CODE
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.INFO)  # Make sure level is set to INFO

        # Get the specific logger used in threshold_selector.py
        self.logger = logging.getLogger('threshold_selector')  # Use the same logger name as in your module
        self.logger.addHandler(self.log_handler)
        self.logger.setLevel(logging.INFO)

        # Initialize the selector with sample data
        self.selector = ThresholdSelector(self.sample_data)
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove log handler
        self.logger.removeHandler(self.log_handler)

        self.log_capture.close()
    
    def test_initialization(self):
        """Test that the ThresholdSelector initializes correctly."""
        # Verify metrics data was stored and sorted
        self.assertEqual(len(self.selector.metrics_data), 5)
        self.assertEqual(self.selector.metrics_data[0].threshold, 0.1)
        self.assertEqual(self.selector.metrics_data[-1].threshold, 0.5)
        
        # Verify metrics were computed
        self.assertEqual(len(self.selector.threshold_metrics), 5)
        self.assertIn(0.1, self.selector.threshold_metrics)
        self.assertIn(0.5, self.selector.threshold_metrics)
        
        # Verify log message for initialization
        self.assertIn("Initialized ThresholdSelector", self.log_capture.getvalue())
    
    def test_validation_empty_data(self):
        """Test validation for empty data."""
        with self.assertRaises(ValueError) as context:
            ThresholdSelector([])
        self.assertIn("No metrics data provided", str(context.exception))
    
    def test_validation_duplicate_thresholds(self):
        """Test validation for duplicate thresholds."""
        duplicate_data = self.sample_data + [ConfusionMatrixMetrics(0.3, 80, 85, 15, 20)]
        with self.assertRaises(ValueError) as context:
            ThresholdSelector(duplicate_data)
        self.assertIn("Duplicate threshold", str(context.exception))
    
    def test_validation_threshold_out_of_bounds(self):
        """Test validation for thresholds outside [0,1]."""
        # Test threshold < 0
        invalid_data_1 = self.sample_data + [ConfusionMatrixMetrics(-0.1, 80, 85, 15, 20)]
        with self.assertRaises(ValueError) as context:
            ThresholdSelector(invalid_data_1)
        self.assertIn("out of bounds", str(context.exception))
        
        # Test threshold > 1
        invalid_data_2 = self.sample_data + [ConfusionMatrixMetrics(1.1, 80, 85, 15, 20)]
        with self.assertRaises(ValueError) as context:
            ThresholdSelector(invalid_data_2)
        self.assertIn("out of bounds", str(context.exception))
    
    def test_validation_negative_counts(self):
        """Test validation for negative counts in confusion matrix."""
        negative_data = self.sample_data + [ConfusionMatrixMetrics(0.6, -5, 97, 3, 25)]
        with self.assertRaises(ValueError) as context:
            ThresholdSelector(negative_data)
        self.assertIn("Negative values", str(context.exception))
    
    def test_calculate_metrics(self):
        """Test the calculation of classification metrics."""
        # Test with normal values
        recall, precision, f1, specificity = self.selector._calculate_metrics(80, 90, 10, 20)
        
        self.assertAlmostEqual(recall, 0.8)
        self.assertAlmostEqual(precision, 0.8888888889)
        self.assertAlmostEqual(f1, 0.8421052632)
        self.assertAlmostEqual(specificity, 0.9)
    
    def test_calculate_metrics_zero_division(self):
        """Test metric calculation with zero denominators."""
        # Test TP+FN = 0 (recall)
        recall, precision, f1, _ = self.selector._calculate_metrics(0, 90, 10, 0)
        self.assertEqual(recall, 0.0)
        
        # Test TP+FP = 0 (precision)
        recall, precision, f1, _ = self.selector._calculate_metrics(0, 90, 0, 10)
        self.assertEqual(precision, 0.0)
        
        # Test precision+recall = 0 (F1)
        recall, precision, f1, _ = self.selector._calculate_metrics(0, 90, 10, 10)
        self.assertEqual(f1, 0.0)
        
        # Test TN+FP = 0 (specificity)
        _, _, _, specificity = self.selector._calculate_metrics(80, 0, 0, 20)
        self.assertEqual(specificity, 0.0)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation using Wilson method."""
        # Test with reasonable sample size
        lower, upper = self.selector._calculate_recall_confidence_interval(90, 10)
        
        # Verify bounds are reasonable and contain the actual recall
        self.assertTrue(0 <= lower <= 0.9)
        self.assertTrue(0.9 <= upper <= 1)
        
        # Test that Wilson CIs are different from normal approximation
        # (This is more of a sanity check than a strict test)
        recall = 90 / (90 + 10)
        se = np.sqrt((recall * (1 - recall)) / (90 + 10))
        normal_lower = max(0, recall - 1.96 * se)
        normal_upper = min(1, recall + 1.96 * se)
        
        # Wilson intervals should be different from normal approximation
        self.assertNotEqual(round(lower, 4), round(normal_lower, 4))
        self.assertNotEqual(round(upper, 4), round(normal_upper, 4))
    
    def test_confidence_interval_edge_cases(self):
        """Test confidence interval calculation with edge cases."""
        # Test with zero denominator
        lower, upper = self.selector._calculate_recall_confidence_interval(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)
        
        # Test with perfect recall
        lower, upper = self.selector._calculate_recall_confidence_interval(100, 0)
        self.assertAlmostEqual(upper, 1.0)
        self.assertTrue(lower < 1.0)  # Wilson interval provides a more conservative lower bound
        
        # Test with zero recall
        lower, upper = self.selector._calculate_recall_confidence_interval(0, 100)
        self.assertAlmostEqual(lower, 0.0, places=15)  # Allow small floating-point differences

        self.assertTrue(upper > 0.0)  # Wilson interval provides a more conservative upper bound
    
    def test_find_best_threshold(self):
        """Test finding the best threshold."""
        # Test with minimum recall = 0.9 (should return 0.2)
        best = self.selector.find_best_threshold(min_recall=0.9)
        self.assertEqual(best, 0.2)
        
        # Test with minimum recall = 0.95 (should return 0.1)
        best = self.selector.find_best_threshold(min_recall=0.95)
        self.assertEqual(best, 0.1)
        
        # Test with minimum recall = 0.96 (should return None)
        best = self.selector.find_best_threshold(min_recall=0.96)
        self.assertIsNone(best)
        
        # Verify logging
        log_output = self.log_capture.getvalue()
        self.assertIn("Finding best threshold", log_output)
        self.assertIn("Selected best threshold", log_output)
    
    def test_find_best_threshold_with_ci(self):
        """Test finding the best threshold using confidence intervals."""
        # The exact result depends on the confidence intervals, which use Wilson method
        # This is more of an integration test to ensure the method runs without errors
        best = self.selector.find_best_threshold(min_recall=0.85, use_confidence_interval=True)
        
        # We expect a more conservative threshold when using CIs
        # But the exact value depends on the specific CI calculation
        self.assertIsNotNone(best)
        
        # Verify logging
        log_output = self.log_capture.getvalue()
        self.assertIn("use_confidence_interval=True", log_output)
    
    def test_invalid_min_recall(self):
        """Test with invalid min_recall values."""
        with self.assertRaises(ValueError) as context:
            self.selector.find_best_threshold(min_recall=-0.1)
        self.assertIn("must be in [0, 1]", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.selector.find_best_threshold(min_recall=1.5)
        self.assertIn("must be in [0, 1]", str(context.exception))
    
    def test_get_metrics_report(self):
        """Test generating metrics report."""
        report = self.selector.get_metrics_report()
        
        self.assertEqual(len(report), 5)
        self.assertIn(0.1, report)
        
        # Check that all expected metrics are present
        metrics = report[0.1]
        required_metrics = ['recall', 'precision', 'f1', 'specificity', 
                           'recall_lower_ci', 'recall_upper_ci', 'accuracy']
        for metric in required_metrics:
            with self.subTest(metric=metric):
                self.assertIn(metric, metrics)

    
    def test_plot_metrics(self):
        """Test the plot_metrics method."""
        # Mock plt.show to avoid displaying the plot during tests
        with patch('matplotlib.pyplot.show') as mock_show:
            self.selector.plot_metrics()
            mock_show.assert_called_once()
        
        # Test saving the plot
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            self.selector.plot_metrics(save_path='test_plot.png')
            mock_savefig.assert_called_once()
    
    def test_export_to_dataframe(self):
        """Test exporting metrics to a pandas DataFrame."""
        df = self.selector.export_to_dataframe()
        
        # Check that the DataFrame has the correct structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        
        # Check that DataFrame is sorted by threshold
        self.assertTrue(df['threshold'].equals(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])))
        
        # Check that all required columns are present
        required_columns = ['threshold', 'recall', 'precision', 'f1', 
                           'specificity', 'recall_lower_ci', 'recall_upper_ci']
        for column in required_columns:
            self.assertIn(column, df.columns)
    
    def test_calculate_optimal_threshold_for_metric(self):
        """Test finding optimal thresholds for different metrics."""
        # Test F1 optimization
        f1_threshold, f1_value = self.selector.calculate_optimal_threshold_for_metric('f1')
        self.assertIsNotNone(f1_threshold)
        self.assertGreaterEqual(f1_value, 0)
        self.assertLessEqual(f1_value, 1)
        
        # Test precision optimization
        prec_threshold, prec_value = self.selector.calculate_optimal_threshold_for_metric('precision')
        self.assertIsNotNone(prec_threshold)
        self.assertGreaterEqual(prec_value, 0)
        self.assertLessEqual(prec_value, 1)
        
        # Test invalid metric
        with self.assertRaises(ValueError) as context:
            self.selector.calculate_optimal_threshold_for_metric('invalid_metric')
        self.assertIn("Metric must be one of", str(context.exception))
    
    def test_integration(self):
        """Test the full workflow in an integrated manner."""
        # Create sample data with known properties
        data = [
            ConfusionMatrixMetrics(0.1, 100, 80, 20, 0),    # Recall: 1.0
            ConfusionMatrixMetrics(0.5, 80, 90, 10, 20),    # Recall: 0.8
            ConfusionMatrixMetrics(0.9, 50, 95, 5, 50),     # Recall: 0.5
        ]
        
        # Create selector
        selector = ThresholdSelector(data)
        
        # Find best threshold with minimum recall 0.9
        best_threshold = selector.find_best_threshold(min_recall=0.9)
        self.assertEqual(best_threshold, 0.1)
        
        # Find optimal F1 threshold
        f1_threshold, _ = selector.calculate_optimal_threshold_for_metric('f1')
        
        # Export to DataFrame
        df = selector.export_to_dataframe()
        self.assertEqual(len(df), 3)
        
        # Verify consistency across all components
        self.assertEqual(df[df['threshold'] == best_threshold]['recall'].iloc[0], 1.0)


if __name__ == '__main__':
    unittest.main()