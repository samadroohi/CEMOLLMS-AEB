import numpy as np
from config import Config
import re
from .base import BaseConformalPredictor

class MulticlassConformalPredictor(BaseConformalPredictor):
    def __init__(self):
        super().__init__()
        self.task_type = "multiclass_classification"
    def fit(self, y_true_calib, pred_calibration,prob_pred_calib, alpha):
        """
        Perform calibration to find the threshold t_alpha for each label
        
        Args:
            y_true_calib: True labels for calibration set. Each element is a list of emotion labels
            prob_pred_calib: Predicted probabilities for calibration set
            alpha: Desired miscoverage level (e.g., 0.1 for 90% coverage)
            
        Returns:
            dict: Threshold t_alpha for each label
        """
        q_hat = {values:[] for values in Config.VALID_D_TYPES[Config.DS_TYPE].values()}
        scores = {values:[] for values in Config.VALID_D_TYPES[Config.DS_TYPE].values()}
        
        for i, labels in enumerate(y_true_calib):
           probs = prob_pred_calib[i]
           if len(probs) > 0:
                for label in labels:
                    #remove any space or special characters
                    label = re.sub(r'[^a-zA-Z]', '', label)
                    #remove spaces and convert to lowercase
                    label = label.strip().lower()
                    label_idx = next((int(key) for key, value in Config.VALID_D_TYPES[Config.DS_TYPE].items() if value == label), None)
                    if label_idx is not None:
                        scores[label.strip()].append(1-max(probs[j][label_idx] for j in range(len(probs))))

        for label in scores:
            if len(scores[label]) > 0:
                # sort scores
                sorted_scores = sorted(scores[label])
                q_hat[label] = np.percentile(sorted_scores, 100 * (1 - alpha))
            else:
                q_hat[label] = 0.5  # default threshold

        return q_hat

    def predict(self, prob_pred_test, q_hat):
        """
        Get prediction sets for test data
        
        Args:
            prob_pred_test: Predicted probabilities for test set
            q_hat: Threshold t_alpha for each label
            
        Returns:
            list: Prediction sets for each sample
        """
        prediction_sets = []
        for probs in prob_pred_test:
            if len(probs) > 0:
                pred_set = []
                for label, label_idx in Config.VALID_D_TYPES[Config.DS_TYPE].items():
                    if label_idx is not None:
                        nonconformity_score = 1 - max(probs[j][int(label)] for j in range(len(probs)))
                        if nonconformity_score <= q_hat[label_idx]:
                            pred_set.append(label)
                prediction_sets.append(pred_set)
            else:
                prediction_sets.append([])
        return prediction_sets

    def get_conformal_results(self, true_labels,pred_test, prob_pred_test, q_hat):
        """
        Get conformal prediction results and statistics
        
        Args:
            true_labels: True labels for the test set
            prob_pred_test: Predicted probabilities for test set
            q_hat: Threshold t_alpha for each label
            
        Returns:
            tuple: (prediction_sets, coverage, avg_set_size, true_labels)
        """
        prediction_sets = self.predict(prob_pred_test, q_hat)
        # Compute coverage
        coverage = self._compute_coverage(prediction_sets, true_labels)
        
        # Compute average set size
        avg_set_size = self._compute_avg_set_size(prediction_sets)
        
        return prediction_sets, coverage, avg_set_size, true_labels

       
    def _compute_coverage(self, prediction_sets, true_labels):
        psets = []
        for labels in true_labels:
            true_indexes = []
            for label in labels:
                true_indexes.append(next((key for key, value in Config.VALID_D_TYPES[Config.DS_TYPE].items() if value == label), None))
            psets.append(true_indexes)
        correct_count = sum(
            set(true_set).issubset(set(pred_set))
            for true_set, pred_set in zip(psets, prediction_sets)
        )
        return correct_count / len(true_labels)

    def _compute_avg_set_size(self, prediction_sets):
        """Helper method to compute average set size"""
        if not prediction_sets:
            return 0.0
        return sum(len(pred_set) for pred_set in prediction_sets) / len(prediction_sets) 