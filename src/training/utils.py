"""Small stateful helpers used by the training loop."""

import torch


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    """Saves the model state whenever validation accuracy improves."""

    def __init__(self, save_path):
        self.save_path = save_path
        self.best_val_acc = 0.0

    def step(self, model, val_acc: float, extra: dict | None = None) -> bool:
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            payload = {"model_state_dict": model.state_dict(), "val_acc": val_acc}
            if extra:
                payload.update(extra)
            torch.save(payload, self.save_path)
            return True
        return False
