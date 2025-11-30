import pytest
from omegaconf import OmegaConf

@pytest.fixture
def base_config():
    """
    Returns a basic Hydra-like DictConfig for testing.
    """
    conf = {
        "pipeline": {
            "type": "torch",
            "model": {"name": "mlp", "hidden_dim": 64},
            "processing": {"name": "raw"},
            "train": {"epochs": 1, "lr": 0.01}
        },
        "wandb_project": "test_project",
        "seed": 42
    }
    return OmegaConf.create(conf)

@pytest.fixture
def kernel_config():
    """
    Returns a config specifically for kernel methods.
    """
    conf = {
        "pipeline": {
            "type": "kernel",
            "model": {"name": "svm", "C": 1.0},
            "execution": {"n_jobs": 1}
        }
    }
    return OmegaConf.create(conf)