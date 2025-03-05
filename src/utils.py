from config import Config
from conformalprediction.regression import ConformalRegressionPredictor
from conformalprediction.classification import ClassificationConformalPredictor
from conformalprediction.multiclass import MulticlassConformalPredictor
from conformalprediction.ordinal_classification import OrdinalClassificationConformalPredictor
import json
import os
import numpy as np
import torch
import re

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
                # Handle both direct float values and strings with numbers followed by text
                if isinstance(result["prediction"], (int, float)):
                    pred = float(result["prediction"])
                else:
                    # Extract the number from the beginning of the string
                    prediction_str = str(result["prediction"]).strip()
                    # Use a regex to extract the first number (integer or decimal)
                    match = re.match(r'^(-?\d+(\.\d+)?)', prediction_str)
                    if match:
                        pred = float(match.group(1))
                    else:
                        stats["invalid_predictions"] += 1
                        continue
                
                if pred >= valid_range["min"] and pred <= valid_range["max"]:
                    stats["valid_predictions"] += 1
                    result["prediction"] = pred
                    valid_results.append(result)
                else:
                    stats["invalid_predictions"] += 1
            except (ValueError, TypeError):
                stats["invalid_predictions"] += 1
                
        except Exception:
            stats["invalid_predictions"] += 1
    
    # Save statistics
    os.makedirs(f'results/statistics/{ds_type}', exist_ok=True)
    stats_file = f'results/statistics/{ds_type}/{model_info}.json'
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
    os.makedirs(f'results/statistics/{ds_type}', exist_ok=True)
    stats_file = f'results/statistics/{ds_type}/{model_info}.json'
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
    if ds_type == "GoEmotions" or ds_type == "E-c":
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
        valid_classes = Config.VALID_D_TYPES[ds_type].values()
        if not valid_classes:
            raise ValueError(f"Unknown dataset type: {ds_type}")
        
        for result in results:
            for each_class in result["prediction"]:
                #remove any space or special characters
                each_class = re.sub(r'[^a-zA-Z]', '', each_class)
                #remove spaces and convert to lowercase
                each_class = each_class.strip().lower()
                if each_class not in valid_classes:
                    stats["invalid_predictions"] += 1
                    break
            stats["valid_predictions"] += 1
            valid_results.append(result)

    
    os.makedirs(f'results/statistics/{ds_type}', exist_ok=True)
    stats_file = f'results/statistics/{ds_type}/{model_info}.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\nPrediction Statistics for {model_info}:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Valid predictions: {stats['valid_predictions']}")
    print(f"Invalid predictions: {stats['invalid_predictions']}")
    print(f"Statistics saved to: {stats_file}")
    
    return valid_results

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

def get_response_multiclass(generated_tokens, logits, tokenizer, ds_type):
    if ds_type == "GoEmotions" or ds_type == "E-c":
        responses = []
        for i, token in enumerate(generated_tokens):    
            answer = tokenizer.decode(token, skip_special_tokens=True).lower().strip()
            if answer == ",":
                continue
            elif answer == ".":
                break
            for value in Config.VALID_D_TYPES[ds_type].values():
                if value.startswith(answer):
                    responses.append(value)
                    break

        return responses

    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    
def get_probs(generated_tokens, logits, tokenizer, ds_type):
    if ds_type =="EI-oc" or ds_type == "SST5" or ds_type == "TDT":
        #0,1,2,3
        #for token in generated_tokens:
            #print(token, tokenizer.decode(token))
            
        for i, token in enumerate(generated_tokens):
            answer = tokenizer.decode(token, skip_special_tokens=True)
            # Extract just the numeric part (remove colons and other characters)

            clean_answer = answer.split(':')[0].strip()
            if clean_answer in Config.VALID_D_TYPES[ds_type].keys():
                if Config.MODEL_NAME_OR_PATH == "lzw1008/Emobloom-7b":  
                    #ADD : to the key
                    encoded_classes = [tokenizer.encode(key + ":", add_special_tokens=False) for key in Config.VALID_D_TYPES[ds_type].keys()]
                else:
                    encoded_classes = [tokenizer.encode(key, add_special_tokens=False) for key in Config.VALID_D_TYPES[ds_type].keys()]
                #for token in encoded_classes:
                 #   print(token, tokenizer.decode(token))
                probs = [float(torch.softmax(logits[i], dim=0)[token[1] if len(token)>1 else token]) for token in encoded_classes]
                
                return probs
    elif ds_type == "V-oc":
        # Handle values: 3,2,1,0,-1,-2,-3
        if Config.MODEL_NAME_OR_PATH == "lzw1008/Emobloom-7b":
            classes = ['3:','2:','1:','0:','-1','-2','-3']
            for i, token in enumerate(generated_tokens):
                answer = tokenizer.decode(token, skip_special_tokens=True)
                cleaned_answer = answer.split(':')[0].strip()
                if cleaned_answer in Config.VALID_D_TYPES[ds_type].keys():
                    encoded_classes = [tokenizer.encode(key, add_special_tokens=False) for key in classes]
                    probs = [float(torch.softmax(logits[i], dim=0)[token[1] if len(token)>1 else token]) for token in encoded_classes]
                    return probs
        else:
            unsigned_classes = ['3','2','1','0']
            unsigned_tokens = [tokenizer.encode(str(key), add_special_tokens=False) for key in unsigned_classes]
            minus_token = tokenizer.encode('-', add_special_tokens=False)[0]  # Get the token ID for minus
            for i, token in enumerate(generated_tokens):
                answer = tokenizer.decode(token, skip_special_tokens=True)
                cleaned_answer = answer.split(':')[0].strip()
                
                if cleaned_answer == '-':
                    # Case: negative number (-3,-2,-1)
                    # Get probabilities for the second token (the number after minus)
                    next_token_logits = logits[i + 1]
                    # Only 3,2,1 can follow the minus sign
                    valid_negative_nums = unsigned_tokens[:-1]  # Exclude '0'
                    
                    # Calculate probabilities for negative numbers
                    negative_probs = []
                    for num_token in valid_negative_nums:
                        token_index = 1 if len(num_token) > 1 else 0
                        # Probability = P(-) * P(number|-)
                        prob = torch.softmax(logits[i], dim=0)[minus_token] * \
                            torch.softmax(next_token_logits, dim=0)[num_token[token_index]]
                        negative_probs.append(float(prob))
                    
                    # Zero probability for unused negative numbers
                    while len(negative_probs) < 3:
                        negative_probs.append(0.0)
                    
                    # Calculate probabilities for positive numbers (including 0)
                    positive_probs = []
                    for num_token in unsigned_tokens:
                        prob = torch.softmax(logits[i], dim=0)[num_token[token_index]]
                        positive_probs.append(float(prob))
                    
                    # Combine probabilities in order [3,2,1,0,-1,-2,-3]
                    probs = positive_probs + negative_probs[::-1]  # Reverse negative probs
                    return probs
                    
                elif cleaned_answer in unsigned_classes:
                    # Case: positive number or zero
                    current_logits = logits[i]
                    
                    # Calculate probabilities for positive numbers (including 0)
                    positive_probs = []
                    for num_token in unsigned_tokens:
                        token_index = 1 if len(num_token) > 1 else 0
                        # Direct probability of the positive number
                        prob = torch.softmax(current_logits, dim=0)[num_token[token_index]]
                        positive_probs.append(float(prob))
                    
                    # Calculate probabilities for negative numbers
                    # P(negative) = P(current_digit) * P(-|digit) * P(second_digit|-)
                    negative_probs = []
                    next_position_logits = logits[i + 1] if i + 1 < len(logits) else None
                    
                    if next_position_logits is not None:
                        for num_token in unsigned_tokens[:-1]:  # Exclude 0 for negative numbers
                            # Probability of seeing minus after the current digit
                            p_minus = torch.softmax(next_position_logits, dim=0)[minus_token]
                            # Probability of seeing the digit after minus
                            p_digit = torch.softmax(next_position_logits, dim=0)[num_token[token_index]]
                            # Joint probability
                            prob = float(p_minus * p_digit)
                            negative_probs.append(prob)
                    else:
                        # If we're at the end of the sequence, negative numbers are impossible
                        negative_probs = [0.0] * 3
                    
                    # Combine probabilities in order [3,2,1,0,-1,-2,-3]
                    probs = positive_probs + negative_probs[::-1]
                    return probs
    elif ds_type == "GoEmotions" or ds_type == "E-c":
        encoded_classes = [tokenizer.encode(key, add_special_tokens=False) for key in Config.VALID_D_TYPES[ds_type].values()]
        if Config.MODEL_NAME_OR_PATH == "lzw1008/Emobloom-7b":
            multiclass_encoded_classes = [tokenizer.encode(" "+key, add_special_tokens=False) for key in Config.VALID_D_TYPES[ds_type].values()]
        probs = []
        multiclass= False
        for i, token in enumerate(generated_tokens):
            answer = tokenizer.decode(token, skip_special_tokens=True).lower().strip()
            cleaned_answer = answer.split(':')[0].strip()
            if cleaned_answer == ".":
                break
                
            # Check if this token is the first token of any class
            is_class_token = False
            for valid_tok in encoded_classes:
                first_token = valid_tok[0] if isinstance(valid_tok, list) and len(valid_tok) > 0 else valid_tok
                if token == first_token:
                    is_class_token = True
                    break
            if Config.MODEL_NAME_OR_PATH == "lzw1008/Emobloom-7b" and multiclass:
                for token_list in multiclass_encoded_classes:
                    if token == token_list[0]:
                        is_class_token = True
                        encoded_classes = multiclass_encoded_classes
                        break
                    
            if is_class_token:
                # Get logits for each class (using first token of each class)
                classes_logits = []
                for token_list in encoded_classes:
                    # Extract the first token from each class encoding
                    class_token_id = token_list[0] if isinstance(token_list, list) and len(token_list) > 0 else token_list
                    class_logit = logits[i][class_token_id].item()
                    classes_logits.append(class_logit)
                # Compute normalized probabilities (softmax over just our classes)
                class_probs_normalized = [float(prob) for prob in torch.softmax(torch.tensor(classes_logits), dim=0).tolist()]
                
                probs.append(class_probs_normalized)
                multiclass = True
                
        return probs
    else:
        print(f"Invalid answer: {cleaned_answer}")
        return None
    


def get_prediction_touples(true_values, predictions,probs, dataset_type):
    '''
    This method generates tuples (key,class) for classification tasks e.g., ("joy", 3)
    Input format example: "3: high amount of joy can be inferred"
    
    Processes both true_values and predictions in parallel, ensuring only pairs
    where both are valid are included in the results.
    
    Args:
        true_values (list): List of ground truth values
        predictions (list): List of model predictions
        dataset_type (str): Type of dataset being processed
        
    Returns:
        tuple: (processed_true_values, processed_predictions)
    '''

    true_result = []
    pred_result = []
    probs_result = []
    
    if dataset_type == "EI-oc":
        emotion_labels = ['anger', 'fear', 'joy', 'sadness']
        
        for true, pred, prob in zip(true_values, predictions,probs):
            try:
                # Process true value
                true_class_index = int(true.split(':')[0].strip())
                true_emotion = None
                for label in emotion_labels:
                    if label in true.lower():
                        true_emotion = label
                        break
                
                # Process prediction
                pred_class_index = int(pred.split(':')[0].strip())
                pred_emotion = None
                for label in emotion_labels:
                    if label in pred.lower():
                        pred_emotion = label
                        break

                
                # Only add if both are valid
                if true_emotion is not None and pred_emotion is not None and prob is not None:
                    true_result.append((true_emotion, true_class_index))
                    pred_result.append((pred_emotion, pred_class_index))
                    probs_result.append(prob)
                
            except (ValueError, AttributeError):
                continue
                
    elif dataset_type == "V-oc" or dataset_type == "SST5" or dataset_type == "TDT":
        for true, pred, prob in zip(true_values, predictions,probs):
            try:
                # Process true value
                true_index = true.strip().split(":")[0].strip()
                true_class_index = list(Config.VALID_D_TYPES[dataset_type].keys()).index(true_index)
                
                # Process prediction
                pred_index = pred.strip().split(":")[0].strip()
                pred_class_index = list(Config.VALID_D_TYPES[dataset_type].keys()).index(pred_index)
                
                # Add valid pairs
                true_result.append((None, true_class_index))
                pred_result.append((None, pred_class_index))
                probs_result.append(prob)
                
            except (ValueError, AttributeError, IndexError):
                continue
                
    elif dataset_type == "GoEmotions" or dataset_type == "E-c":
        for true, pred, prob in zip(true_values, predictions,probs):
            try:
                # Process true value
                if isinstance(true, str):
                    true_list = true.strip().split(",")
                    true_list = [re.sub(r'[^a-zA-Z]', '', each_class) for each_class in true_list]
                    true_list = [each_class.strip().lower() for each_class in true_list]
                elif isinstance(true, list):
                    true_list = true
                else:
                    continue
                
                # Process prediction
                if isinstance(pred, str):
                    pred_list = pred.strip().split(",")
                    pred_list = [re.sub(r'[^a-zA-Z]', '', each_class) for each_class in pred_list]
                    pred_list = [each_class.strip().lower() for each_class in pred_list]
                elif isinstance(pred, list):
                    pred_list = pred
                else:
                    continue
                
                # Validate against valid classes
                valid_classes = set(Config.VALID_D_TYPES[dataset_type].values())
                true_list = [c for c in true_list if c in valid_classes]
                pred_list = [c for c in pred_list if c in valid_classes]
                
                # Only add if both have valid classes
                if true_list and pred_list:
                    true_result.append(true_list)
                    pred_result.append(pred_list)
                    probs_result.append(prob)
            except (ValueError, AttributeError):
                continue
    
    return true_result, pred_result, probs_result
