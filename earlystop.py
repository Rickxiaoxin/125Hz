import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc = np.Inf
        self.delta = delta

    def __call__(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model)
            self.counter = 0

    def save_checkpoint(self, acc, model):
        if self.verbose:
            print(
                f"Validation acc increased ({self.acc:.6f} --> {acc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"checkpoint{acc}.pth")
        self.acc = acc
