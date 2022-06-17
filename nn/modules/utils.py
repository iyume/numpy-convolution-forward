from nn import functional as F

from .module import Module


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)
