import pytest
from unittest.mock import MagicMock, patch
from src.pipelines.base_pipeline import BasePipeline

# Create a concrete class for testing purposes
class TestablePipeline(BasePipeline):
    def load_data(self): pass
    def build_model(self): pass
    def train(self): pass
    def evaluate(self): pass
    
    # We mock this to track if it gets called
    def _save_implementation(self, path):
        pass

@patch('src.pipelines.base_pipeline.wandb') # Mock wandb inside the base_pipeline module
def test_save_checkpoint_workflow(mock_wandb, base_config):
    """
    Test that calling save_checkpoint:
    1. Determines the path using wandb.run.dir
    2. Calls the subclass _save_implementation
    3. Creates a wandb Artifact
    4. Logs the artifact
    """
    # 1. Setup the Mock Environment
    mock_wandb.run.dir = "/tmp/wandb/run-123"
    mock_wandb.run.id = "run-123"
    
    # Instantiate our concrete test class
    pipeline = TestablePipeline(base_config)
    
    # Spy on the _save_implementation method
    # (This lets us see if it was called without overriding it)
    with patch.object(pipeline, '_save_implementation') as mock_impl:
        
        # 2. Execute the method under test
        pipeline.save_checkpoint("my_model")
        
        # 3. Assertions
        
        # A. Was the path constructed correctly?
        expected_path = "/tmp/wandb/run-123/my_model"
        mock_impl.assert_called_once_with(expected_path)
        
        # B. Was a WandB Artifact created?
        mock_wandb.Artifact.assert_called_once_with(
            name="model-run-123", 
            type="model"
        )
        
        # C. Was the file added to the artifact?
        # Get the mock artifact instance returned by the constructor
        mock_artifact_instance = mock_wandb.Artifact.return_value
        mock_artifact_instance.add_file.assert_called_once_with(expected_path)
        
        # D. Was log_artifact called?
        mock_wandb.log_artifact.assert_called_once_with(mock_artifact_instance)