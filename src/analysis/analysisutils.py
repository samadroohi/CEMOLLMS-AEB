import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def analyze_regression_results(results, ds_type, task_types):
    """
    Analyze regression results from conformal prediction output.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        task_types (dict): Dictionary mapping task types to dataset types
        
    Returns:
        dict: Dictionary containing regression metrics
    """
    try:
        if not isinstance(results, dict):
            raise ValueError("Results must be a dictionary")
            
        # Check if we have the required keys
        required_keys = ["ds_type", "true_values", "predictions"]
        if not all(key in results for key in required_keys):
            raise ValueError("Results file missing required data")
            
        # filter results using DS_TYPE
        if results["ds_type"] != ds_type:
            raise ValueError(f"No results found for DS_TYPE: {ds_type}")
            
        if ds_type in task_types["regression"]:
            # Convert strings to float if necessary and create numpy arrays
            true_values = np.array([float(x) for x in results["true_values"]], dtype=np.float64)
            predictions = np.array([float(x) for x in results["predictions"]], dtype=np.float64)
            
            # Calculate regression metrics
            pearson_corr, p_value = stats.pearsonr(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            r2 = r2_score(true_values, predictions)
            
            metrics = {
                'pearson_correlation': pearson_corr,
                'p_value': p_value,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            return metrics
            
    except Exception as e:
        raise Exception(f"Error analyzing regression results: {str(e)}")

def analyze_classification_results(results, ds_type, task_types):
    """
    Analyze binary classification results from conformal prediction output.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        task_types (dict): Dictionary mapping task types to dataset types
        
    Returns:
        dict: Dictionary containing classification metrics
    """
    pass

def analyze_multiclass_results(results, ds_type, task_types):
    """
    Analyze multiclass classification results from conformal prediction output.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        task_types (dict): Dictionary mapping task types to dataset types
        
    Returns:
        dict: Dictionary containing multiclass classification metrics
    """
    pass

def analyze_results(results, ds_type, task_types):
    """
    Main analysis function that routes to specific analysis based on dataset type.
    
    Args:
        results (dict): Dictionary containing prediction results
        ds_type (str): Dataset type
        task_types (dict): Dictionary mapping task types to dataset types
        
    Returns:
        dict: Dictionary containing appropriate metrics for the task type
    """
    if ds_type in task_types["regression"]:
        return analyze_regression_results(results, ds_type, task_types)
    elif ds_type in task_types["classification"]:
        return analyze_classification_results(results, ds_type, task_types)
    elif ds_type in task_types["multiclass_classification"]:
        return analyze_multiclass_results(results, ds_type, task_types)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

def regression_calibration_diagram(results: dict,
                                 ds_type: str,
                                 alpha: float,
                                 output_dir: str = None,
                                 title: str = None,
                                 figsize: tuple = (10, 6)):
    """
    Create a calibration diagram for regression predictions showing prediction intervals
    and whether true values fall within them.
    
    Args:
        results (dict): Dictionary containing conformal prediction results
        ds_type (str): Dataset type
        alpha (float): Significance level (alpha value)
        output_dir (str, optional): Directory to save the plot
        title (str, optional): Custom title for the plot
        figsize (tuple, optional): Figure size (width, height)
    """
    try:
        # Extract and convert data from results to float arrays
        y_true = np.array([float(x) for x in results["true_values"]], dtype=np.float64)
        y_pred = np.array([float(x) for x in results["predictions"]], dtype=np.float64)
        
        # Handle prediction sets that come as separate lower and upper bounds
        if isinstance(results["prediction_sets"], list) and len(results["prediction_sets"]) == 2:
            lower_bounds = np.array([float(x) for x in results["prediction_sets"][0]], dtype=np.float64)
            upper_bounds = np.array([float(x) for x in results["prediction_sets"][1]], dtype=np.float64)
        else:
            raise ValueError("Prediction sets must be a list containing [lower_bounds, upper_bounds]")
        
        # Sort all arrays by predicted values for better visualization
        sort_idx = np.argsort(y_pred)
        y_pred = y_pred[sort_idx]
        y_true = y_true[sort_idx]
        lower_bounds = lower_bounds[sort_idx]
        upper_bounds = upper_bounds[sort_idx]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot prediction intervals
        x_range = np.arange(len(y_pred))
        plt.fill_between(x_range, lower_bounds, upper_bounds, 
                        alpha=0.2, color='blue', label='Prediction interval')
        
        # Plot regression line
        z = np.polyfit(x_range, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(x_range, p(x_range), 'orange', label='Regression line')
        
        # Plot points with colors based on whether true value is in interval
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        
        # Plot points inside interval
        plt.scatter(x_range[in_interval], y_true[in_interval], 
                   color='green', marker='o', label='Within interval', alpha=0.6)
        
        # Plot points outside interval
        plt.scatter(x_range[~in_interval], y_true[~in_interval], 
                   color='red', marker='x', label='Outside interval', alpha=0.6)
        
        # Calculate coverage
        coverage = np.mean(in_interval)
        interval_sizes = upper_bounds - lower_bounds
        avg_interval_size = np.mean(interval_sizes)
        
        # Add title and labels
        if title is None:
            title = f'Regression Calibration ({ds_type}, α={alpha})\n' \
                    f'Coverage: {coverage:.3f}, Avg Interval Size: {avg_interval_size:.3f}'
        plt.title(title)
        plt.xlabel('Instance Index (sorted by predicted value)')
        plt.ylabel('Target Value')
        
        # Add legend
        plt.legend()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'calibration_plot_{ds_type}_alpha_{alpha}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        return {
            'coverage': coverage,
            'avg_interval_size': avg_interval_size,
            'in_interval_count': np.sum(in_interval),
            'total_points': len(y_true)
        }
    
    except Exception as e:
        print(f"Error in regression_calibration_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"predictions type: {type(results['predictions'][0])}")
        print(f"prediction_sets type: {type(results['prediction_sets'][0][0])}")
        raise
    