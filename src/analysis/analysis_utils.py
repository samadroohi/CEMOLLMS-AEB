import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score

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
    # Extract predictions and true values
    y_true = results['true_values']
    y_pred = results['predictions']
    
    # Calculate Jaccard index score (jacS)
    jaccard_scores = []
    for true, pred in zip(y_true, y_pred):
        true_set = set(true)
        pred_set = set(pred)
        
        if not true_set and not pred_set:  # Both empty
            jaccard_scores.append(1.0)
        elif not true_set or not pred_set:  # One empty, one not
            jaccard_scores.append(0.0)
        else:
            # Jaccard index = intersection/union
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            jaccard_scores.append(intersection / union)
    
    jaccard_index = sum(jaccard_scores) / len(jaccard_scores)
    
    # Get unique labels from both predictions and true values
    unique_labels = set()
    for labels in y_true + y_pred:
        unique_labels.update(labels)
    unique_labels = sorted(list(unique_labels))
    
    # Convert to binary matrix format for F1 scores
    def to_binary_matrix(label_sets, unique_labels):
        matrix = np.zeros((len(label_sets), len(unique_labels)))
        for i, label_set in enumerate(label_sets):
            for label in label_set:
                j = unique_labels.index(label)
                matrix[i, j] = 1
        return matrix
    
    y_true_binary = to_binary_matrix(y_true, unique_labels)
    y_pred_binary = to_binary_matrix(y_pred, unique_labels)
    
    # Calculate F1 scores
    f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro')
    f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro')
    
    return {
        'jaccard_index': jaccard_index,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }

def analyze_ordinal_classification_results(results, ds_type, task_types):
    """
    Analyze ordinal classification results following SemEval-2018 Task1 metrics:
    - Pearson correlation for regression tasks
    - Accuracy, micro-F1, macro-F1 for ordinal classification
    """
    try:
        if not isinstance(results, dict):
            raise ValueError("Results must be a dictionary")
            
        if ds_type == "EI-oc":
            # Group predictions by emotion
            emotion_groups = {}
            
            for true_val, pred_val in zip(results["true_values"], results["predictions"]):
                true_emotion, true_intensity = true_val
                pred_emotion, pred_intensity = pred_val
                
                if true_emotion not in emotion_groups:
                    emotion_groups[true_emotion] = {
                        'true_intensities': [],
                        'pred_intensities': []
                    }
                
                # Always append true intensity
                emotion_groups[true_emotion]['true_intensities'].append(int(true_intensity))
                # If emotion matches, use predicted intensity; if not, use 0
                pred_int = int(pred_intensity) if true_emotion == pred_emotion else 0
                emotion_groups[true_emotion]['pred_intensities'].append(pred_int)
            
            # Calculate metrics for each emotion
            emotion_pearson = {}
            all_true = []
            all_pred = []
            
            for emotion, data in emotion_groups.items():
                true_ints = np.array(data['true_intensities'])
                pred_ints = np.array(data['pred_intensities'])
                
                # Calculate Pearson correlation
                pearson = np.corrcoef(true_ints, pred_ints)[0, 1]
                emotion_pearson[emotion] = float(pearson)
                
                all_true.extend(true_ints)
                all_pred.extend(pred_ints)
            
            # Convert to numpy arrays for overall metrics
            all_true = np.array(all_true)
            all_pred = np.array(all_pred)
            
            # Calculate accuracy
            accuracy = metrics.accuracy_score(all_true, all_pred)
            # Calculate micro and macro F1
            micro_f1 = metrics.f1_score(all_true, all_pred, average='micro')
            macro_f1 = metrics.f1_score(all_true, all_pred, average='macro')
            # Calculate macro-average (average of per-emotion Pearson correlations)
            macro_average = np.mean(list(emotion_pearson.values()))
            average_pearson, _ = stats.pearsonr(all_true, all_pred)

            metrics_dict = {
                'accuracy': float(accuracy),
                'micro_f1': float(micro_f1),
                'macro_f1': float(macro_f1),
                'macro_average': float(macro_average),
                'per_emotion_pearson': emotion_pearson,
                'average_pearson':average_pearson
            }
        elif ds_type == "V-oc" or ds_type == "SST5" or ds_type == "TDT":
            # Convert predictions and true values to numeric values
            true_values = np.array([x[1] for x in results["true_values"]], dtype=np.int32)
            predictions = np.array([x[1] for x in results["predictions"]], dtype=np.int32)
            if ds_type == "V-oc":
                # Map class indices to actual values: [3,2,1,0,-1,-2,-3]
                value_mapping = {0: 3, 1: 2, 2: 1, 3: 0, 4: -1, 5: -2, 6: -3}
                true_values = np.array([value_mapping[x] for x in true_values])
                predictions = np.array([value_mapping[x] for x in predictions])
            elif ds_type == "TDT":
                value_mapping = {0:1, 1:0, 2:-1} 
                true_values = np.array([value_mapping[x] for x in true_values])
                predictions = np.array([value_mapping[x] for x in predictions])
            
            # Calculate metrics
            pearson_corr, p_value = stats.pearsonr(true_values, predictions)
            accuracy = metrics.accuracy_score(true_values, predictions)
            micro_f1 = metrics.f1_score(true_values, predictions, average='micro')
            macro_f1 = metrics.f1_score(true_values, predictions, average='macro')
            
            metrics_dict = {
                'accuracy': float(accuracy),
                'micro_f1': float(micro_f1),
                'macro_f1': float(macro_f1),
                'macro_average': None,  # Not applicable for V-oc since there's only one dimension
                'per_emotion_pearson': None,  # Not applicable for V-oc
                'average_pearson': float(pearson_corr)  # This is our main correlation metric for V-oc
            }
            
        return metrics_dict
            
    except Exception as e:
        raise Exception(f"Error analyzing ordinal classification results: {str(e)}")

def get_performance_metrics(results, ds_type, task_types):
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
    elif ds_type in task_types["multiclass_classification"] :
        return analyze_multiclass_results(results, ds_type, task_types)
    elif ds_type in task_types["ordinal_classification"]:
        return analyze_ordinal_classification_results(results, ds_type, task_types)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

