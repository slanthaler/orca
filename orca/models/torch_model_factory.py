from omegaconf import OmegaConf
import pprint

from .linear import LinearModel
from .lstm import LSTMModel

_MODEL_REGISTRY = {
    "linear": LinearModel,
    "lstm": LSTMModel,
}

def get_model_class(name: str):
    """
    Retrieves a model class from the registry by name.
    Raises ValueError if not found.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def register_model(name: str, model_class):
    """
    Allows dynamic registration of new models (e.g. from plugins)
    """
    _MODEL_REGISTRY[name] = model_class