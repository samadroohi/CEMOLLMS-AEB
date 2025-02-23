from config import Config
from conformalprediction.regression import ConformalRegressionPredictor
from conformalprediction.classification import ClassificationConformalPredictor
from conformalprediction.multiclass import MulticlassConformalPredictor
from conformalprediction.ordinal_classification import OrdinalClassificationConformalPredictor
import json
import os
import numpy as np
import torch

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
    elif ds_type in Config.TASK_TYPES["ordinal_classification"]:
        return OrdinalClassificationConformalPredictor()

def cleaning_results_regression(results, ds_type):
    """
    Clean results based on dataset type and save statistics.
    
    Args:
        results (list): List of result dictionaries containing predictions
        ds_type (str): Dataset type
        
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
    results_path = Config.RESULTS_FILE
    model_info = os.path.basename(results_path).replace('.json', '')
    
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

def cleaning_results_ordinal_classification(results, ds_type):
    
    # Initialize counters
    stats = {
        "total_predictions": len(results),
        "valid_predictions": 0,
        "invalid_predictions": 0
    }
    
    valid_results = []
    
    # Get model info from the results filename
    results_path = Config.RESULTS_FILE
    model_info = os.path.basename(results_path).replace('.json', '')
    
    # Get valid classes for this dataset type
    valid_classes = Config.VALID_D_TYPES[ds_type].keys()
    if not valid_classes:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    
    # Process each result
    for result in results:
        try:
            # Check if result is properly formatted and has prediction
            if not isinstance(result, dict) or "prediction" not in result:
                stats["invalid_predictions"] += 1
                continue
            
            # Extract the numeric part from the prediction and check if it's in valid classes
            prediction = result["prediction"].strip().split(":")[0].strip()
            if prediction in valid_classes:
                stats["valid_predictions"] += 1
                valid_results.append(result)
            else:
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

def cleaning_results_multiclass_classification(results,ds_type):
    pass

def cleaning_results_classification(results,ds_type):
    pass

def cleaning_results(results,ds_type):
    if ds_type in Config.TASK_TYPES["regression"]:
        return cleaning_results_regression(results,ds_type)
    elif ds_type in Config.TASK_TYPES["classification"]:
        return cleaning_results_classification(results,ds_type)
    elif ds_type in Config.TASK_TYPES["multiclass_classification"]:
        return cleaning_results_multiclass_classification(results,ds_type)
    elif ds_type in Config.TASK_TYPES["ordinal_classification"]:
        return cleaning_results_ordinal_classification(results,ds_type)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


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


def get_probs(generated_tokens, logits, tokenizer, ds_type):
    if ds_type in Config.TASK_TYPES["ordinal_classification"]:
        for i, token in enumerate(generated_tokens):
            answer = tokenizer.decode(token, skip_special_tokens=True)
            if answer in  Config.VALID_D_TYPES[ds_type].keys():
                encoded_classes = [tokenizer.encode(key, add_special_tokens=False) for key in Config.VALID_D_TYPES[ds_type].keys()]
                probs = [logits[i][token[1]] for token in encoded_classes]
                return probs
            
        print(f"Invalid answer: {answer}")
        return None
def get_prediction_touples(predictions, dataset_type):
    '''
    This method generates tuple (key,class) for classification tasks e.g., ("joy", 3)
    Input format example: "3: high amount of joy can be inferred"
    '''
    result = []
    if dataset_type == "EI-oc":
        emotion_labels = ['anger', 'fear', 'joy', 'sadness']
        
        for pred in predictions:
            try:
                # Extract the numeric class (e.g., "3" from "3: high amount of joy...")
                class_index = int(pred.split(':')[0].strip())
                
                # Find which emotion is mentioned in the prediction
                emotion = None
                for label in emotion_labels:
                    if label in pred.lower():
                        emotion = label
                        break
                
                if emotion is not None:
                    result.append((emotion, class_index))
                
            except (ValueError, AttributeError):
                continue
                
    return result
