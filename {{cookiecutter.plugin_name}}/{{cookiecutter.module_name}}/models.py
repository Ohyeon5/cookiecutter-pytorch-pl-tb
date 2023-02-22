from typing import Callable, Optional
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



class SimpleNet(nn.Module):
    def __init__(
        self, input_size: int = 2, hidden_size: int = 2, n_layers: int = 1
    ) -> None:
        super(SimpleNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        for _ in range(n_layers):
            self.encoder.append(nn.Linear(hidden_size, hidden_size))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class LitSimpleNet(pl.LightningModule):
    def __init__(
        self,
        net: Optional[SimpleNet] = None,
        input_size: int = 2,
        hidden_size: int = 2,
        n_layers: int=1,
        loss_fn: Callable = F.binary_cross_entropy_with_logits,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.net = (
            SimpleNet(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers)
            if net is None
            else net
        )
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(x))

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss)

