from .base import BaseConformalPredictor

class OrdinalClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(self):
        super().__init__()
        self.task_type = "ordinal_classification"
    
    def fit(self, y_true, y_pred, alpha):
        pass
    
    def predict(self, y_pred, threshold):
        pass

    def get_conformal_results(self, y_true, y_pred, threshold):
        pass
    
    def evaluate_coverage(self, prediction_sets, true_labels):
        pass
    def evaluate_set_size(self, prediction_sets, precision=3):
        pass