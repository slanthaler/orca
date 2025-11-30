from sklearn.preprocessing import StandardScaler
import torch

def get_preprocessor(processing_cfg):
    """
    Returns a processing function or sklearn transformer 
    based on the Hydra config.
    """
    method = processing_cfg.name
    params = processing_cfg.params

    if method == "raw":
        # Return an identity function or a simple pass-through class
        return None 
    
    elif method == "standard_scaler":
        return StandardScaler(
            with_mean=params.with_mean, 
            with_std=params.with_std
        )
        
    else:
        raise ValueError(f"Unknown processing method: {method}")