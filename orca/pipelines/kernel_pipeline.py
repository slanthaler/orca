import os
import wandb
from omegaconf import OmegaConf

from .base_pipeline import BasePipeline, _print_chapter
# Assuming you might have a kernel factory or just use sklearn directly
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class KernelPipeline(BasePipeline):
    
    def __init__(self, config, eval_config):
        super().__init__(config, eval_config)
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        _print_chapter("Loading Data (Kernel)")
        # Load matrices here
        pass

    def build_model(self):
        _print_chapter("Building Model (Kernel)")
        model_args = OmegaConf.to_container(self.config.pipeline.model, resolve=True)
        name = model_args.pop("name")
        
        if name == "svm":
            self.model = SVC(**model_args)
        elif name == "rf":
            self.model = RandomForestClassifier(**model_args)
        else:
            raise ValueError(f"Unknown kernel model: {name}")

    def build_trainer(self):
        # Kernel methods usually don't need a separate optimizer setup step
        # But this is a good place to set random states or specialized CV splitters
        pass

    def train(self):
        _print_chapter("Training (Fitting)")
        # self.model.fit(self.X_train, self.y_train)
        print("Model fitted.")

    def evaluate(self):
        _print_chapter("Evaluating")
        # acc = self.model.score(self.X_test, self.y_test)
        acc = 0.85 # Dummy
        metrics = {"accuracy": acc}
        wandb.log(metrics)
        return metrics

    def save_checkpoint(self, name="model_checkpoint", dir=None, **kwargs):
        pass