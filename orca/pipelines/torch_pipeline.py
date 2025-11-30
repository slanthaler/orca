from pprint import pprint
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf

from .base_pipeline import BasePipeline, _print_chapter
from ..models.torch_model_factory import get_model_class
from ..trainer import build_trainer
from ..datasets import np_data_to_torch_data


class TorchPipeline(BasePipeline):
    def __init__(self, config, eval_config):
        super().__init__(config, eval_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.data = None
        
    def process_data(self):
        self.data = np_data_to_torch_data(
            self._data_raw, 
            self.config.training.batch_size
        )

    def build_model(self):
        _print_chapter('Building Model')
        model_cfg = OmegaConf.to_container(self.config.model, resolve=True)
        model_type = model_cfg.pop("type")
        print('Building model type:', model_type)
        print('Model config:')
        pprint(model_cfg)
        #
        model_class = get_model_class(model_type)
        self.model = model_class(
            input_dim=self.data['input_dim'], 
            output_dim=self.data['output_dim'],
            **model_cfg,
        )
        print(f"Model built: {self.model}")
        
    def build_trainer(self):
        _print_chapter('Building Trainer')
        print('Training config:')
        pprint(dict(self.config.training))
        self.trainer = build_trainer(
            model=self.model,
            train_loader=self.data['train_loader'],
            val_loader=self.data['val_loader'],
            device=self.device,
            config=self.config.training,
        )

    def train(self):
        self.trainer.fit()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            y_pred_tensor = self.model(x_tensor)
            y_pred = y_pred_tensor.cpu().numpy()
        return y_pred

    def save_checkpoint(self, name = "model_checkpoint", dir = None):
        dir = dir or self.run_dir
        checkpoint_path = dir / f"{name}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")