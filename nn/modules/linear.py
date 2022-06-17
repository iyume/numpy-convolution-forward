import numpy as np

from .module import Module
from nn import functional as F


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weight = np.zeros((out_features, in_features))
        self.bias = np.zeros(out_features)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
