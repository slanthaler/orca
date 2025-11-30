import wandb
import numpy as np
from .metrics import get_metric_function

class Evaluator:
    def __init__(self, metric_names: list):
        """
        Args:
            metric_names: List of strings, e.g. ["mse", "mae"] or ["accuracy"]
        """
        self.metric_names = metric_names
        
    def evaluate(self, y_true, y_pred, prefix="val"):
        """
        Calculates metrics and returns a dictionary.
        
        Args:
            y_true: Numpy array of ground truth
            y_pred: Numpy array of predictions
            prefix: String prefix for logging (e.g. "val", "test")
        """
        results = {}
        
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        for name in self.metric_names:
            metric_fn = get_metric_function(name)
            
            try:
                score = metric_fn(y_true, y_pred)
                key = f"{prefix}_{name}"
                results[key] = score
            except ValueError as e:
                print(f"Skipping metric {name}: {e}")

        return results