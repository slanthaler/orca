import pytest
from orca.pipelines.base_pipeline import BasePipeline

def test_cannot_instantiate_base_pipeline(base_config):
    """
    Test that we cannot create an instance of BasePipeline directly.
    """
    with pytest.raises(TypeError) as excinfo:
        BasePipeline(base_config)
    
    # Check that the error message mentions abstract methods
    assert "Can't instantiate abstract class" in str(excinfo.value)

def test_cannot_instantiate_incomplete_subclass(base_config):
    """
    Test that a subclass failing to implement abstract methods 
    raises an error.
    """
    class BrokenPipeline(BasePipeline):
        def load_data(self): pass
        def build_model(self): pass
        def train(self): pass
        def evaluate(self): pass
        # Oops! We forgot _save_implementation
    
    with pytest.raises(TypeError) as excinfo:
        BrokenPipeline(base_config)
    
    assert "_save_implementation" in str(excinfo.value)

def test_can_instantiate_complete_subclass(base_config):
    """
    Test that a correctly implemented subclass works.
    """
    class ValidPipeline(BasePipeline):
        def load_data(self): pass
        def build_model(self): pass
        def train(self): pass
        def evaluate(self): pass
        def _save_implementation(self, path): pass
        
    pipeline = ValidPipeline(base_config)
    assert pipeline is not None
    assert pipeline.cfg == base_config