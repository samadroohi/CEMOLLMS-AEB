import numpy as np
from abc import ABC, abstractmethod

class BaseConformalPredictor(ABC):
    def __init__(self):
        self.task_type = None
        self.classes = None
    
    @abstractmethod
    def fit(self, y_true, y_pred, alpha):
        pass
    
    @abstractmethod
    def predict(self, y_pred, quantiles):
        pass
    
    @abstractmethod
    def get_conformal_results(self, y_true, y_pred, quantiles):
        pass 