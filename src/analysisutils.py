import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def analyze_regression_metrics(true_values, predictions):
    """
    Calculate various regression metrics including Pearson correlation coefficient.
    
    Args:
        true_values (array-like): Ground truth values
        predictions (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing various regression metrics
    """
    # Convert inputs to numpy arrays
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    # Calculate Pearson correlation coefficient
    pearson_corr, p_value = stats.pearsonr(true_values, predictions)
    
    # Calculate other regression metrics
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    
    # Calculate mean and std of prediction intervals if available
    metrics = {
        "pearson_correlation": pearson_corr,
        "p_value": p_value,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    
    return metrics
