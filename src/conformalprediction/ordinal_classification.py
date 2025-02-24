import torch
import numpy as np
from typing import List, Optional
from .base import BaseConformalPredictor

class OrdinalClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(self):
        super().__init__()
        self.task_type = "classification"
    
    def fit(self, y_true, y_pred,probs_calibration, alpha):
        """
        Calibrate the conformal predictor for classification.
        
        Args:
            y_true: True class labels (numeric indices)
            y_pred: Softmax probabilities of shape (n_samples, n_classes)
            probs_calibration: logit vector of the ordinal classes
            alpha: Significance level
            
        Returns:
            Conformity threshold
        """
        # Convert inputs to tensors
        softmax_probabilities = torch.tensor(probs_calibration, dtype=torch.float32)
        true_labels = torch.tensor([touple[1] for touple in y_true], dtype=torch.long)

        # Compute nonconformity scores
        conformity_scores = 1 - softmax_probabilities[torch.arange(len(true_labels)), true_labels]
        sorted_scores = torch.sort(conformity_scores)[0]

        # Compute threshold
        n = len(sorted_scores)
        index = int(np.ceil((n + 1) * (1 - alpha))) - 1
        index = max(0, min(index, n - 1))
        
        return sorted_scores[index].item()

    def predict(self, y_pred, probs_test, threshold):
        """
        Get prediction sets for each sample.
        
        Args:
            y_pred: Softmax probabilities
            threshold: Conformity threshold from calibration
            
        Returns:
            List of prediction sets for each sample
        """
        softmax_probabilities = torch.tensor(probs_test, dtype=torch.float32)
        prediction_sets = []
        for probs in softmax_probabilities:
            nonconformity_scores = 1 - probs
            pred_set = torch.where(nonconformity_scores <= threshold)[0].tolist()
            prediction_sets.append(pred_set)
            
        return prediction_sets

    def get_conformal_results(self, y_true, y_pred,probs_test, threshold):
        """
        Get conformal prediction results and statistics.
        
        Args:
            y_true: True class labels
            y_pred: Softmax probabilities
            threshold: Conformity threshold
            
        Returns:
            tuple: (prediction_sets, coverage, avg_set_size, y_true)
        """
        true_labels = [touple[1] for touple in y_true]
        predictions= [touple[1] for touple in y_pred]
        prediction_sets = self.predict(predictions, probs_test,threshold)
        
        # Calculate coverage
        coverage = self.evaluate_coverage(prediction_sets, true_labels)
        
        # Calculate average set size
        avg_set_size = self.evaluate_set_size(prediction_sets)
        
        return prediction_sets, coverage, avg_set_size, y_true

    def evaluate_coverage(self, prediction_sets: List[List[int]], true_labels) -> float:
        if isinstance(true_labels[0], str):
            numeric_labels = []
            for label in true_labels:
                try:
                    num = int(label.split(':')[0].strip())
                    numeric_labels.append(num)
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Unable to extract numeric label from '{label}'") from e
            true_labels = torch.tensor(numeric_labels, dtype=torch.long)
        elif not isinstance(true_labels, torch.Tensor):
            true_labels = torch.tensor(true_labels, dtype=torch.long)

        covered = sum(int(true_label) in pred_set 
                     for true_label, pred_set in zip(true_labels, prediction_sets))
        return covered / len(true_labels)

    def evaluate_set_size(self, prediction_sets: List[List[int]], precision: Optional[int] = 3) -> float:
        if len(prediction_sets) == 0:
            return 0.0

        avg_size = sum(len(pred_set) for pred_set in prediction_sets) / len(prediction_sets)
        return round(avg_size, precision) if precision is not None else avg_size 