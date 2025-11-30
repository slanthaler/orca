import torch
import wandb
from tqdm import tqdm


def build_trainer(model, train_loader, val_loader, device, config):
    """
    Factory function to build and return a BasicTrainer instance.

    Args:
        model: The PyTorch model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: torch.device
        config: The 'training' section of the hydra config
    """
    criterion = None
    if config.loss == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss}")

    optimizer = None
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    trainer = BasicTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    return trainer      


class BasicTrainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        val_loader=None, 
        device=None,
        config=None
    ):
        """
        Args:
            model: The PyTorch model
            optimizer: The PyTorch optimizer
            criterion: Loss function (e.g. MSELoss)
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
            device: torch.device
            config: The 'train' section of the hydra config (epochs, logging, etc.)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = config

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        # Use tqdm for a progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
                
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(x)
            
            # Compute loss
            loss = self.criterion(pred, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Optional: Log step-wise metrics
            if self.cfg.log_steps and batch_idx % self.cfg.log_interval == 0:
                wandb.log({"train_step_loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self, epoch):
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return {"val_loss": avg_loss}

    def fit(self):
        """
        Main entry point to start training.
        """
        print(f"Starting training on {self.device} for {self.cfg.epochs} epochs.")
        self.model.to(self.device)
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Construct log dictionary
            log_dict = {
                "epoch": epoch + 1,
                "train_epoch_loss": train_loss,
                **val_metrics
            }
            
            # Print and Log
            max_width = len(str(self.cfg.epochs))
            print(f"Epoch {epoch+1:{max_width}}: Train Loss {train_loss:10.4f} | Val Loss {val_metrics.get('val_loss', None):10.4f}")
            wandb.log(log_dict)