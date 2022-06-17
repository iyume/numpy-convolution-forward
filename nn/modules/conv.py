from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt

from nn import functional as F

from .module import Module


_size = Tuple[int, int]

_pair: Callable[[Union[int, _size]], _size] = (
    lambda x: (x, x) if isinstance(x, int) else x
)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, _size],
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        self.kernel_size: _size = kernel_size
        self.stride: _size = stride
        self.padding: _size = padding
        self.weight: npt.NDArray[np.float64] = np.zeros(
            (out_channels, in_channels, *kernel_size)
        )
        self.bias: npt.NDArray[np.float64] = np.zeros(out_channels)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return F.conv2d(
            x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding
        )
