import numpy as np
from torch.nn import Module
class EarlyStopping:
    def __init__(self, patience: int = 1, min_delta: float = 0) -> None:
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.best_model_state = None
        
    def early_stop(self, val_loss: float, model: Module) -> bool:
        
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False