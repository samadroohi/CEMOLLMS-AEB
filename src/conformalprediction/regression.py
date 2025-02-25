from .base import BaseConformalPredictor
import numpy as np

class ConformalRegressionPredictor(BaseConformalPredictor):
    def __init__(self):
        super().__init__()
        self.task_type = "regression"
        self.residuals = None

    def fit(self, y_true, y_pred,probs_calibration, alpha):
        """
        Computes the baseline conformal prediction quantiles.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            alpha: Significance level (e.g., 0.1 for 90% confidence)
            
        Returns:
            tuple: (lower_quantile, upper_quantile)
        """
        # Convert inputs to float arrays
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # Compute residuals (not absolute)
        self.residuals = y_true - y_pred
        
        # Calculate lower and upper quantiles
        n = len(self.residuals)
        
        # Calculate asymmetric quantiles
        lower_q = np.quantile(self.residuals, alpha/2, interpolation='higher')
        upper_q = np.quantile(self.residuals, 1 - alpha/2, interpolation='higher')
        
        return (lower_q, upper_q)
    
    def predict(self, y_pred,probs_test, quantiles):
        lower_q, upper_q = quantiles
        # Convert predictions to float array
        y_pred = np.array(y_pred, dtype=float)
        
        lower_bounds = y_pred + lower_q
        upper_bounds = y_pred + upper_q

        return lower_bounds, upper_bounds
    
    def get_conformal_results(self, y_true, y_pred,probs_test, quantiles):
        """
        Get conformal prediction results and statistics.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            quantiles: Tuple of (lower_quantile, upper_quantile)
            
        Returns:
            tuple: ((lower_bounds, upper_bounds), coverage, interval_size, y_true)
        """
        # Convert inputs to float arrays
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # Get both bounds from predict
        lower, upper = self.predict(y_pred,None, quantiles)
        
        # Check coverage
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        interval_size = np.mean(upper - lower)
        
        print(f"\nResults:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Average interval size: {interval_size:.4f}")
        
        return (lower, upper), coverage, interval_size, y_true
        