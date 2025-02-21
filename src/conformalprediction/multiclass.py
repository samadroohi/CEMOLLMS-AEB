import numpy as np

from .base import BaseConformalPredictor

class MulticlassConformalPredictor(BaseConformalPredictor):
    def __init__(self):
        super().__init__()
        self.task_type = "multiclass_classification"
        
    def fit(self, y_true_calib, prob_pred_calib, alpha):
        """
        Perform calibration to find the threshold t_alpha for each label
        
        Args:
            y_true_calib: True labels for calibration set. Each element is a list of emotion labels
            prob_pred_calib: Predicted probabilities for calibration set
            alpha: Desired miscoverage level (e.g., 0.1 for 90% coverage)
            
        Returns:
            dict: Threshold t_alpha for each label
        """
        #q_hat = {keys:[] for keys in DATA_CONFIG["data_bound"].keys()}
        #scores = {keys:[] for keys in DATA_CONFIG["data_bound"].keys()}
        
        #for i, labels in enumerate(y_true_calib):
         #   probs = []
          #  if len(prob_pred_calib[i]) > 0:
           #     for prob in prob_pred_calib[i]:
           #         probs.append(self._softmax(prob))
                #for label in labels:
                 #   label_idx = DATA_CONFIG["data_bound"][label.strip()]
                  #  if label_idx is not None:
                   #     scores[label.strip()].append(1-max(probs[j][label_idx] for j in range(len(probs))))

        #for label in scores:
         #   if len(scores[label]) > 0:
          #      sorted_scores = sorted(scores[label])
           #     q_hat[label] = np.percentile(sorted_scores, 100 * (1 - alpha))
            #else:
             #   q_hat[label] = 0.5  # default threshold

        #return q_hat

    def predict(self, prob_pred_test, q_hat):
        """
        Get prediction sets for test data
        
        Args:
            prob_pred_test: Predicted probabilities for test set
            q_hat: Threshold t_alpha for each label
            
        Returns:
            list: Prediction sets for each sample
        """
        #prediction_sets = []
        #for probs in prob_pred_test:
        #    if len(probs) > 0:
        #        probs = [self._softmax(prob) for prob in probs]
        #        pred_set = []
        #        for label, label_idx in DATA_CONFIG["data_bound"].items():
        #            if label_idx is not None:
        #                nonconformity_score = 1 - max(probs[j][label_idx] for j in range(len(probs)))
        #                if nonconformity_score <= q_hat[label]:
        #                    pred_set.append(label)
        #        prediction_sets.append(pred_set)
        #    else:
        #        prediction_sets.append([])
        return []

    def get_conformal_results(self, true_labels, prob_pred_test, q_hat):
        """
        Get conformal prediction results and statistics
        
        Args:
            true_labels: True labels for the test set
            prob_pred_test: Predicted probabilities for test set
            q_hat: Threshold t_alpha for each label
            
        Returns:
            tuple: (prediction_sets, coverage, avg_set_size, true_labels)
        """
        #prediction_sets = self.predict(prob_pred_test, q_hat)
        # Compute coverage
        #coverage = self._compute_coverage(prediction_sets, true_labels)
        
        # Compute average set size
        #avg_set_size = self._compute_avg_set_size(prediction_sets)
        
        return [], 0, 0, true_labels

    #def _softmax(self, x):
    #    """Helper method to compute softmax"""
    #    exp_x = np.exp(x)
    #    return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _compute_coverage(self, prediction_sets, true_labels):
        """Helper method to compute coverage"""
        correct_count = sum(
            set(true_set).issubset(set(pred_set))
            for true_set, pred_set in zip(true_labels, prediction_sets)
        )
        return correct_count / len(true_labels)

    def _compute_avg_set_size(self, prediction_sets):
        """Helper method to compute average set size"""
        if not prediction_sets:
            return 0.0
        return sum(len(pred_set) for pred_set in prediction_sets) / len(prediction_sets) 