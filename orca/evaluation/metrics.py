import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

def get_metric_function(name):
    """
    Returns a metric function (y_true, y_pred) -> float
    """
    REGISTRY = {
        # Regression
        "mse": mean_squared_error,
        "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error,
        "r2": lambda y, y_pred: 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    }
    
    if name not in REGISTRY:
        raise ValueError(f"Metric '{name}' not found. Available: {list(REGISTRY.keys())}")
        
    return REGISTRY[name]