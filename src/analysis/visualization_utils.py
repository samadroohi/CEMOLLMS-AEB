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
    (Modified to compute average bin confidence)
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
        
        # Define uniform bins manually
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []  # average predicted probability for each bin
        bin_counts = []
        
        # Loop over bins
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            bin_count = np.sum(mask)
            
            if bin_count >= 10:  # Only compute when there are sufficient samples
                # Compute bin accuracy (fraction of correct predictions)
                pred_labels = np.array([pred[1] for pred in results["predictions"]])
                bin_accuracy = np.mean(y_true[mask] == pred_labels[mask])
                # Compute average confidence of predictions in this bin
                avg_confidence = np.mean(predictions[mask])
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
                bin_counts.append(bin_count)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', alpha=0.5)
        
        # Only plot bins with enough data
        valid_bins = ~np.isnan(bin_accuracies)
        # For display purposes, you might still use the bin centers:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.plot(bin_centers[valid_bins], bin_accuracies[valid_bins], 'b-', label='Model calibration')
        plt.scatter(bin_centers[valid_bins], bin_accuracies[valid_bins], c='blue')
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        if title is None:
            title = f'Reliability Diagram ({dataset_type})'
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Frequency')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        
        # Compute ECE using the actual average predicted probabilities
        total_samples = np.sum(bin_counts[valid_bins])
        if total_samples > 0:
            ece = np.sum(bin_counts[valid_bins] * np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])) / total_samples
            
            # Calculate Maximum Calibration Error (mcale)
            calibration_errors = np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])
            mcale = np.max(calibration_errors) if len(calibration_errors) > 0 else np.nan
        else:
            ece = np.nan
            mcale = np.nan

        # Annotate the plot with ECE in red text
        plt.text(0.05, 0.90, f'ECE: {ece:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        #Annotate the plot with MCE in red text
        plt.text(0.05, 0.85, f'MCE: {mcale:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        # Other metrics (Brier)
        y_true_one_hot = np.array([pred[1] == y_true[i] for i, pred in enumerate(results["predictions"])])
        brier_score = np.mean((predictions - y_true_one_hot) ** 2)
        
        metrics = {
            'accuracy': float(np.mean(y_true == np.array([pred[1] for pred in results["predictions"]])),
            ),
            'ece': float(ece),
            'mcale': float(mcale),  # Added Maximum Calibration Error
            'brier_score': float(brier_score),
            'mean_prediction': float(np.mean(predictions))
        }
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'reliability_plot_{dataset_type}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
            
        return metrics
        
    except Exception as e:
        print(f"Error in classification_relibaility_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise

def multiclass_classification_relibaility_diagram(results: dict,
                                      dataset_type: str,
                                      output_dir: str = None,
                                      title: str = None,
                                      figsize: tuple = (10, 6)):
    """
    Create a reliability diagram for multilabel classification using Jaccard similarity.
    
    Args:
        results (dict): Dictionary containing prediction results
        dataset_type (str): Type of dataset (e.g., 'E-c')
        output_dir (str, optional): Directory to save the plot
        title (str, optional): Custom title for the plot
        figsize (tuple, optional): Figure size (width, height)
    """
    try:
        # Extract true values and predicted probabilities
        true_values = results["true_values"]
        probs = results["probs"]
        
        # Number of bins for the reliability diagram
        n_bins = 10
        
        # Lists to store confidence and accuracy
        confidences = []
        jaccard_scores = []
        
        # For each example
        for i in range(len(true_values)):
            true_set = set(true_values[i])
            
            # Get predicted probabilities and create predictions
            instance_probs = probs[i]
            predicted_labels = []
            avg_confidence = 0
            
            # For each instance in the example, get prediction with highest prob
            for label_probs in instance_probs:
                if len(label_probs) > 0:  # Ensure we have probabilities
                    class_idx = np.argmax(label_probs)
                    prob = label_probs[class_idx]
                    avg_confidence += prob
                    
                    # Map index to emotion label
                    emotion_labels = ["anger", "disgust", "fear", "joy", "love", 
                                    "optimism", "pessimism", "sadness", "surprise", 
                                    "trust", "neutralornoemotion", "anticipation"]
                    predicted_labels.append(emotion_labels[class_idx])
            
            if len(instance_probs) > 0:
                avg_confidence /= len(instance_probs)
                
            predicted_set = set(predicted_labels)
            
            # Calculate Jaccard similarity
            if len(true_set) == 0 and len(predicted_set) == 0:
                jaccard = 1.0
            elif len(true_set) == 0 or len(predicted_set) == 0:
                jaccard = 0.0
            else:
                jaccard = len(true_set.intersection(predicted_set)) / len(true_set.union(predicted_set))
                
            confidences.append(avg_confidence)
            jaccard_scores.append(jaccard)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Define uniform bins manually
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        # Loop over bins
        for i in range(n_bins):
            mask = (np.array(confidences) >= bin_edges[i]) & (np.array(confidences) < bin_edges[i + 1])
            bin_count = np.sum(mask)
            
            if bin_count >= 10:  # Only compute when there are sufficient samples
                # Compute bin accuracy (average Jaccard similarity)
                bin_accuracy = np.mean(np.array(jaccard_scores)[mask])
                # Compute average confidence of predictions in this bin
                avg_confidence = np.mean(np.array(confidences)[mask])
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
                bin_counts.append(bin_count)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', alpha=0.5)
        
        # Only plot bins with enough data
        valid_bins = ~np.isnan(bin_accuracies)
        # For display purposes, you might still use the bin centers:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.plot(bin_centers[valid_bins], bin_accuracies[valid_bins], 'b-', label='Model calibration')
        plt.scatter(bin_centers[valid_bins], bin_accuracies[valid_bins], c='blue')
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        if title is None:
            title = f'Reliability Diagram ({dataset_type})'
        plt.title(title)
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Jaccard Similarity')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        
        # Compute ECE using the actual average predicted probabilities
        total_samples = np.sum(bin_counts[valid_bins])
        if total_samples > 0:
            ece = np.sum(bin_counts[valid_bins] * np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])) / total_samples
            
            # Calculate Maximum Calibration Error (mcale)
            calibration_errors = np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])
            mcale = np.max(calibration_errors) if len(calibration_errors) > 0 else np.nan
        else:
            ece = np.nan
            mcale = np.nan

        # Annotate the plot with ECE in red text
        plt.text(0.05, 0.90, f'ECE: {ece:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        # Annotate the plot with MCE in red text
        plt.text(0.05, 0.85, f'MCE: {mcale:.3f}', transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), color='red')
        
        # Calculate average Jaccard similarity
        avg_jaccard = np.mean(jaccard_scores)
        
        metrics = {
            'jaccard_similarity': float(avg_jaccard),
            'ece': float(ece),
            'mcale': float(mcale),
            'mean_confidence': float(np.mean(confidences))
        }
        
        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'reliability_plot_{dataset_type}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
            
        return metrics
        
    except Exception as e:
        print(f"Error in multiclass_classification_relibaility_diagram: {str(e)}")
        print("Data types:")
        print(f"true_values type: {type(results['true_values'][0])}")
        print(f"probs type: {type(results['probs'][0])}")
        raise

def cp_diagrams(results,dataset_type, output_dir):
    plot_confidence_vs_coverage(results, dataset_type, output_dir)
    cp_results = plot_coverage_vs_prediction_set_size(results, dataset_type, output_dir)
    return cp_results


    


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
    mcove = 0
    alphas = Config.CP_ALPHA
    for alpha in alphas:
        cov = results[str(alpha)]["coverage"]
        coverage.append(cov)
        ace += abs(cov - (1-alpha))
        mcove = max(mcove, abs(cov - (1-alpha)))
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
    plt.text(0.05, 0.90, f'MCovE: {mcove:.3f}', transform=plt.gca().transAxes, 
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
    Plots prediction set size vs confidence level (1 - alpha) with circles for each point,
    connects the points with a line, and annotates the coverage for each point.
    
    Args:
        results (dict): Dictionary containing conformal prediction results.
        dataset_type (str): Dataset type.
        output_dir (str, optional): Directory to save the plot.
    """
    alphas = Config.CP_ALPHA
    plt.figure(figsize=(8, 6))
    confidences = [1 - alpha for alpha in alphas]
    set_sizes = []
    coverages = []
    cp_metrics = {}
    mcove = 0
    for alpha in alphas:
        result = results[str(alpha)]
        sizes = []
        coverage = result["coverage"]
        mcove = max(mcove, abs(coverage - (1-alpha)))
        for pred_set in result["prediction_sets"]:
            sizes.append(len(pred_set))
        avg_size = np.mean(sizes)
        set_sizes.append(avg_size)
        coverages.append(coverage)
        plt.scatter(1 - alpha, avg_size, c='b', marker='o')
        plt.annotate(f"{coverage:.2f}", (1 - alpha, avg_size), xytext=(5, 5), textcoords="offset points")
    ace = np.mean(np.abs(np.array(coverages) - np.array(confidences)))
    plt.plot(confidences, set_sizes, 'b-')

    plt.xlabel('Confidence Level (1 - alpha)')
    plt.ylabel('Prediction Set Size')
    plt.title(f'Prediction Set Size vs Confidence Level ({dataset_type})')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'size_vs_confidence_{dataset_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    cp_metrics = {
        'alpha': alphas,
        'coverage': coverages,
        'psize': set_sizes,
        'ace': ace,
        'mcove': mcove
    }
    return cp_metrics
        


    
    return metrics
def compute_regression_metrics(y_true, y_pred, model=None, X=None):
    """
    Compute standard regression metrics including uncertainty estimates.
    
    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        model: Trained model (optional) - used to estimate prediction uncertainty
        X: Feature data corresponding to y_true (optional) - needed for uncertainty estimation
        
    Returns:
        dict: Dictionary containing regression metrics
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Calculate basic metrics
    metrics = {
        'mse': np.mean(residuals**2),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals))
    }

    return metrics
def calibration_anlaysis(results, ds_type, output_dir=None):

    if ds_type in Config.TASK_TYPES["ordinal_classification"]:
        calibration_metrics = classification_relibaility_diagram(results[str(Config.CP_ALPHA[0])], ds_type, output_dir =output_dir)
        cp_metrics =  cp_diagrams(results, ds_type, output_dir)
    elif ds_type in Config.TASK_TYPES["regression"]:
        cp_metrics = {'confidences':[],  'coverages': [], 'average_interval_sizes': []}
        for alpha in Config.CP_ALPHA:
            cp_data = regression_calibration_diagram(results[str(alpha)], ds_type, alpha, output_dir =output_dir)
            cp_metrics['confidences'].append(1-alpha)
            cp_metrics['coverages'].append(cp_data['coverage'])
            cp_metrics['average_interval_sizes'].append(cp_data['avg_interval_size'])
            
        calibration_metrics = compute_regression_metrics(results[str(Config.CP_ALPHA[0])]["true_values"], results[str(Config.CP_ALPHA[0])]["predictions"])
    elif ds_type in Config.TASK_TYPES["multiclass_classification"]:
        calibration_metrics = multiclass_classification_relibaility_diagram(results[str(Config.CP_ALPHA[0])], ds_type, output_dir)

    return calibration_metrics, cp_metrics