from typing import Optional
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset


class DataDist(Dataset):
    def __init__(
        self,
        dist: np.ndarray,
        target: np.ndarray = None
    ) -> None:
        self.dist = dist
        self.target = target
        super().__init__()

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.FloatTensor(self.dist[index, :]), torch.FloatTensor([self.target[index]])

    def __len__(self):
        return len(self.dist)

