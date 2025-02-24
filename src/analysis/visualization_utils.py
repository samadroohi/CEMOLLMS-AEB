from matplotlib import pyplot as plt
import os
import numpy as np
from config import Config

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
        data ={
            'coverage': coverage,
            'avg_interval_size': avg_interval_size,
            'in_interval_count': np.sum(in_interval),
            'total_points': len(y_true)
        }
        return data
    
    except Exception as e:
        print(f"Error in regression_calibration_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"predictions type: {type(results['predictions'][0])}")
        print(f"prediction_sets type: {type(results['prediction_sets'][0][0])}")
        raise

def classification_relibaility_diagram(results: dict,
                                     dataset_type: str,
                                     output_dir: str = None,
                                     title: str = None,
                                     figsize: tuple = (10, 6),
                                     ):
    """
    Create a reliability diagram for classification predictions.
    
    Args:
        results (dict): Dictionary containing predictions, true labels, and logits
        dataset_type (str): Dataset type
        alpha (float): Not used in this context, kept for API consistency
        output_dir (str, optional): Directory to save the plot
        title (str, optional): Custom title for the plot
        figsize (tuple, optional): Figure size (width, height)
        debug (bool, optional): Whether to print debug information
    """
    try:

        # Extract true class indices
        y_true = np.array([label[1] for label in results["true_values"]])
        
        
        # Convert logits to probabilities and get probability for predicted class
        predictions = []
        for i, (probs, true_idx) in enumerate(zip(results["probs"], y_true)):
            probs = np.array(probs)
            predictions.append(probs[true_idx])
            
        predictions = np.array(predictions)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Calculate reliability curve with confidence intervals
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accuracies = []
        confidence_intervals = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            bin_count = np.sum(mask)
            
            if bin_count >= 10:  # Only include bins with sufficient samples
                bin_accuracy = np.mean(y_true[mask] == np.array([pred[1] for pred in results["predictions"]])[mask])
                # Calculate confidence interval using Wilson score interval
                z = 1.96  # 95% confidence
                n = bin_count
                p = bin_accuracy
                ci = z * np.sqrt((p * (1-p) + z*z/(4*n)) / n) / (1 + z*z/n)
                
                bin_accuracies.append(bin_accuracy)
                confidence_intervals.append(ci)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(np.nan)
                confidence_intervals.append(np.nan)
                bin_counts.append(bin_count)
        
        bin_accuracies = np.array(bin_accuracies)
        confidence_intervals = np.array(confidence_intervals)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', alpha=0.5)
        
        # Plot reliability curve with confidence intervals
        valid_bins = ~np.isnan(bin_accuracies)
        plt.plot(bin_centers[valid_bins], bin_accuracies[valid_bins], 'b-', label='Model calibration')
        plt.scatter(bin_centers[valid_bins], bin_accuracies[valid_bins], c='blue')
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        if title is None:
            title = f'Reliability Diagram ({dataset_type})'
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        
        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper left')
        
        # Calculate additional metrics
        # ECE (Expected Calibration Error)
        valid_accuracies = bin_accuracies[valid_bins]
        valid_centers = bin_centers[valid_bins]
        valid_counts = np.array(bin_counts)[valid_bins]
        total_samples = np.sum(valid_counts)
        
        # Calculate ECE as weighted average of |accuracy - confidence|
        ece = np.sum(valid_counts * np.abs(valid_accuracies - valid_centers)) / total_samples
        
        # MCE (Maximum Calibration Error)
        mce = np.max(np.abs(valid_accuracies - valid_centers))
        
        # Brier Score (mean squared error between predictions and actual outcomes)
        y_true_one_hot = np.array([pred[1] == y_true[i] for i, pred in enumerate(results["predictions"])])
        brier_score = np.mean((predictions - y_true_one_hot) ** 2)
        
        # Save metrics to file if output_dir is provided
        if output_dir:
            metrics = {
                'accuracy': float(np.mean(y_true == np.array([pred[1] for pred in results["predictions"]]))),
                'ece': float(ece),
                'mce': float(mce),
                'brier_score': float(brier_score),
                'mean_prediction': float(np.mean(predictions))
            }
            metrics_path = os.path.join(output_dir, f'calibration_metrics_{dataset_type}.txt')
            with open(metrics_path, 'w') as f:
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
            print(f"Metrics saved to: {metrics_path}")
        
        # Save or show plot

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'reliability_plot_{dataset_type}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
            
        return metrics
        
    except Exception as e:
        print(f"Error in classification_reliability_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise

def multiclass_classification_relibaility_diagram(results,dataset_type, alpha, output_dir):
    pass
def cp_diagrams(results,dataset_type, output_dir):
    plot_confidence_vs_coverage(results, dataset_type, output_dir)
    plot_coverage_vs_prediction_set_size(results, dataset_type, output_dir)

    


def plot_confidence_vs_coverage(results, dataset_type, output_dir=None):
    """
    Plots confidence vs empirical coverage and annotates the plot with the ACE (Average Coverage Error).
    
    Args:
        results (dict): Dictionary containing conformal prediction results.
        dataset_type (str): Dataset type.
        alphas (list): A list of confidence levels to evaluate.
        output_dir (str, optional): Directory to save the plot.
    """
    coverage = []
    ace = 0
    alphas = Config.CP_ALPHA
    for alpha in alphas:
        cov = results[str(alpha)]["coverage"]
        coverage.append(cov)
        ace += abs(cov - (1-alpha))
    ace /= len(alphas)
    confidence_values = [1 - alpha for alpha in alphas]
    
    plt.figure(figsize=(10, 6))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', alpha=0.5)
    
    # Plot confidence vs coverage
    plt.plot(confidence_values, coverage, 'b-', label='Model calibration')
    plt.scatter(confidence_values, coverage, c='blue')
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.title(f'Confidence vs Empirical Coverage ({dataset_type})')
    plt.xlabel('Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Add annotation for ACE
    plt.text(0.05, 0.95, f'ACE: {ace:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
    
    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'confidence_vs_coverage_{dataset_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
 

def plot_coverage_vs_prediction_set_size(results, dataset_type, output_dir=None):
    """
    Plots coverage vs prediction set size for each alpha.
    
    Args:
        results (dict): Dictionary containing conformal prediction results.
        dataset_type (str): Dataset type.
        alphas (list): A list of confidence levels to evaluate.
        output_dir (str, optional): Directory to save the plot.
    """
    alphas = Config.CP_ALPHA
    plt.figure(figsize=(8, 6))
    for alpha in alphas:
        sizes = []
        coverage = []
        for pred_set in results["prediction_sets"]:
            sizes.append(len(pred_set))
            correct_count = 0
            for label in results["true_values"]:
                if label in pred_set:
                    correct_count += 1
            coverage.append(correct_count / len(results["true_values"]))
        plt.plot(sorted(sizes), sorted(coverage), label=f'Alpha = {alpha}')
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Empirical Coverage')
    plt.title(f'Coverage vs Prediction Set Size ({dataset_type})')
    plt.legend()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'coverage_vs_size_{dataset_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
def calibration_anlaysis(results, ds_type, output_dir=None):
    cp_results = None 
    if ds_type in Config.TASK_TYPES["ordinal_classification"]:
        calibration_metrics = classification_relibaility_diagram(results[str(Config.CP_ALPHA[0])], ds_type, output_dir =output_dir)
        cp_results =  cp_diagrams(results, ds_type, output_dir)
    elif ds_type in Config.TASK_TYPES["regression"]:
        calibration_metrics = regression_calibration_diagram(results, ds_type,  output_dir =output_dir)
    return calibration_metrics, cp_results