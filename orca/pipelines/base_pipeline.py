from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import os 
from pathlib import Path

from ..datasets import default_dataloader
from ..evaluation import Evaluator 

# helper function
def _print_chapter(title):
    print('\n' + '='*10 + f' {title} ' + '='*10)


class BasePipeline(ABC):
    def __init__(self, pipeline_cfg: DictConfig, eval_cfg: DictConfig, run_name: str=None):
        """
        Args:
            pipeline_cfg: The specific config for this pipeline (e.g. cfg.pipeline)
                          It contains .model, .train, .processing
            eval_cfg:     The shared evaluation config (e.g. cfg.evaluation)
                          It contains .metrics, .prefix
        """
        self.config = pipeline_cfg
        self.eval_config = eval_cfg
        self.run_name = run_name
        self._data_raw = None
    
    def prepare_run(self):
        """
        Any preparation steps before running the pipeline.
        """
        _print_chapter('Preparing Run')
        self._setup_run_name_dir()
        print(f"Run name: {self.run_name}")
        print(f"Run directory created at: {self.run_dir}")
        self._dump_config()

    def _setup_run_name_dir(self):
        self.run_name = self.run_name or self._unique_run_name()
        self._check_valid_run_name(self.run_name)
        self.run_dir = Path(__file__).parents[2] / 'runs' / self.run_name
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=False)

    def _dump_config(self):
        config_path = self.run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.config, resolve=True))
        print(f"Config saved to {config_path}")

    def _unique_run_name(self):
        import uuid
        import datetime
        datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{datetime_now}_{uuid.uuid4().hex[:8]}"
    
    def _check_valid_run_name(self, name):
        invalid_chars = set('/\\?%*:|"<>')
        if any((char in invalid_chars) for char in name):
            raise ValueError(f"Invalid run name '{name}'. It cannot contain any of {invalid_chars}")
        
    def run(self):
        """
        High-level method to run the full pipeline.
        """
        self.prepare_run()
        self.load_data()
        self.process_data()
        self.build_model()
        self.build_trainer()
        self.train()
        metrics = self.evaluate()
        self.save_final()
        return metrics

    def load_data(self):
        _print_chapter('Loading Data')
        print(f"Loading data with batch size {self.config.training.batch_size}")
        self._data_raw = default_dataloader() # this loads numpy arrays!

    def evaluate(self):
        """
        Evaluates the model.
        Returns:
            dict: A dictionary of metrics (e.g. {'test_corr': 0.95, 'test_loss': 0.1})
        """
        evaluator = Evaluator(self.eval_config.metrics)
        metrics = {}
        for split in ['train', 'val', 'test']:
            print(f'Evaluating on {split} set...')
            x, y_true = self._data_raw[split]
            y_pred = self.predict(x)
            metrics[split] = evaluator.evaluate(y_true, y_pred, prefix=split)

        self._dump_metrics(metrics)
        return metrics

    def _dump_metrics(self, metrics):   
        import json
        metrics_path = self.run_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            f.write(json.dumps(metrics, indent=4))
        print(f"Metrics saved to {metrics_path}")

    def save_final(self):
        _print_chapter('Saving Final Model')
        self.save_checkpoint(name="final_model")

    @abstractmethod
    def predict(self, x):
        """
        Generates predictions for input data x.
        
        Args:
            x: Input data as numpy array
        Returns:
            Numpy array of predictions
        """
        pass

    @abstractmethod
    def process_data(self):
        """
        Continues preparing data from self._data_raw (according to config.data).
        Torch: Sets up data loaders
        Kernel: Sets up data matrices
        """
        pass

    @abstractmethod
    def build_model(self):
        """
        Initializes the model architecture (config.model)
        """
        pass

    @abstractmethod
    def build_trainer(self):
        """
        Sets up the training logic (config.training)
        """
        pass

    @abstractmethod
    def train(self):
        """
        Executes the training logic (config.training)
        """
        pass

    @abstractmethod
    def save_checkpoint(self, name: str="model_checkpoint", dir: str=None):
        """
        Implementation-specific saving logic.
        Args:
            name (str): The name of the checkpoint file (excluding extension).
            dir (str): Directory to save the checkpoint. If None, uses default self.run_dir.
        """
        pass

