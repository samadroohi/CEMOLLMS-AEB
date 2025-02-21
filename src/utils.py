from config import Config
from conformalprediction.regression import ConformalRegressionPredictor
from conformalprediction.classification import ClassificationConformalPredictor
from conformalprediction.multiclass import MulticlassConformalPredictor
import json
import os
import numpy as np

def compute_probs(ds_type, token_logits):
    if ds_type in Config.CLASSIFICATION_DS_TYPES:
        pass
    elif ds_type in Config.MULTICLASS_CLASSIFICATION_DS_TYPES:
        pass
def get_predictor(ds_type):
    if ds_type in Config.TASK_TYPES["classification"]:
        return ClassificationConformalPredictor()
    elif ds_type in Config.TASK_TYPES["multiclass_classification"]:
        return MulticlassConformalPredictor()
    elif ds_type in Config.TASK_TYPES["regression"]:
        return ConformalRegressionPredictor()

def cleaning_results(results):
    """
    Clean results based on dataset type and save statistics.
    
    Args:
        results (list): List of result dictionaries containing predictions
        
    Returns:
        list: Cleaned results with valid entries only
    """
    # Initialize counters
    stats = {
        "total_predictions": len(results),
        "valid_predictions": 0,
        "invalid_predictions": 0
    }
    
    valid_results = []
    
    # Get model info from the results filename
    results_path = Config.RESULTS_FILE  # This should be the full path to your results file
    filename = os.path.basename(results_path)
    model_info = filename.replace('.json', '')  # e.g., "Emollama-chat-7b_temp0.9"
    
    # Get ds_type from first valid result
    ds_type = None
    for result in results:
        if isinstance(result, dict) and "ds_type" in result:
            ds_type = result["ds_type"]
            break
    
    if not ds_type:
        raise ValueError("Could not determine dataset type from results")
    
    # Get valid range for this dataset type
    valid_range = Config.VALID_D_TYPES.get(ds_type)
    if not valid_range:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    
    # Process each result
    for result in results:
        try:
            # Check if result is properly formatted and has prediction
            if not isinstance(result, dict) or "prediction" not in result:
                stats["invalid_predictions"] += 1
                continue
                
            # Try to convert prediction to float and validate range
            try:
                pred = float(result["prediction"])
                if pred >= valid_range["min"] and pred <= valid_range["max"]:
                    stats["valid_predictions"] += 1
                    valid_results.append(result)
                else:
                    stats["invalid_predictions"] += 1
            except (ValueError, TypeError):
                stats["invalid_predictions"] += 1
                
        except Exception:
            stats["invalid_predictions"] += 1
    
    # Save statistics
    os.makedirs('results/statistics', exist_ok=True)
    stats_file = f'results/statistics/{model_info}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\nPrediction Statistics for {model_info}:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Valid predictions: {stats['valid_predictions']}")
    print(f"Invalid predictions: {stats['invalid_predictions']}")
    print(f"Statistics saved to: {stats_file}")
    
    return valid_results
def save_cp_results(dataset_type, input_test, true_test, pred_test, probs_test, conformal_results, alpha):
    path = Config.CONFORMAL_RESULTS_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Helper function to safely convert numpy arrays and values
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    try:
        # For the first alpha value (0.1), start fresh by clearing existing results
        if alpha == Config.CP_ALPHA[0]:
            existing_results = {}
        else:
            # For subsequent alpha values, load existing results
            with open(path, 'r') as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Could not read existing results file. Starting fresh.")
                    existing_results = {}

        # Convert prediction_sets tuple to list first
        prediction_sets = list(conformal_results[0]) if isinstance(conformal_results[0], tuple) else conformal_results[0]
        
        # Create results for current alpha
        alpha_results = {
            "ds_type": dataset_type,
            "alpha": float(alpha),
            "coverage": convert_to_serializable(conformal_results[1]),
            "true_values": convert_to_serializable(true_test),
            "predictions": convert_to_serializable(pred_test),
            "probs": convert_to_serializable(probs_test),
            "prediction_sets": convert_to_serializable(prediction_sets),
            "interval_size": convert_to_serializable(conformal_results[2])
        }
        
        # Add or update results for this alpha
        existing_results[str(alpha)] = alpha_results
        
        # Save updated results
        with open(path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print(f"Conformal Prediction Results for α={alpha} saved to: {path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


        