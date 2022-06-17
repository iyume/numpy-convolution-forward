from typing import Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided


_size = Tuple[int, int]


def pad(
    input: npt.NDArray[np.float64],
    pad: Tuple[Tuple[int, int], ...],
    mode: Literal["constant"] = "constant",
    value: float = 0,
):
    """
    TODO:
        1. reflect pad
        2. other padding mode support...
    """
    if mode != "constant":
        raise ValueError("not implemented")
    return np.pad(input, pad, mode, constant_values=value)  # type: ignore


def conv2d(
    input: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    bias: Optional[npt.NDArray[np.float64]] = None,
    stride: Union[int, _size] = 1,
    padding: Union[int, _size] = 0,
) -> npt.NDArray[np.float64]:
    """Conv2d over an input image.

    Args:
        input: image (N,C,H,W)
        weight: kernel (out_channels,in_channels,kH,kW)
        bias: bias for each out_channel (out_channels)
        stride: int or tuple(sH,sW)
        padding: int or tuple(padH,padW)

    TODO:
        1. dialation implement
        2. groups implement
    """
    if input.ndim != 4:
        raise ValueError(f"input must be (N,C,H,W), got {input.shape}")
    elif weight.ndim != 4:
        raise ValueError(f"weight must be 4 dimensions, got {weight.ndim}")
    elif input.shape[1] != weight.shape[1]:
        raise ValueError(
            "input channels must be equal to weight's in_channels, "
            f"got input {input.shape} and weight {weight.shape}"
        )
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    H_out = (
        (input.shape[2] + 2 * padding[0] - 1 * (weight.shape[2] - 1) - 1) // stride[0]
    ) + 1
    W_out = (
        (input.shape[3] + 2 * padding[1] - 1 * (weight.shape[3] - 1) - 1) // stride[1]
    ) + 1
    # change input size
    input = pad(
        input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
    )
    if weight.shape[2:] > input.shape[2:]:
        raise ValueError(
            "weight's kernel size cannot be greater than input image size, "
            f"got input {input.shape} and weight {weight.shape}"
        )
    output = np.zeros((input.shape[0], weight.shape[0], H_out, W_out))
    for out_channel in range(weight.shape[0]):
        for index in np.ndindex(H_out, W_out):
            anchor = index[0] * stride[0], index[1] * stride[1]
            subarray = input[
                :,
                :,
                anchor[0] : anchor[0] + weight.shape[2],
                anchor[1] : anchor[1] + weight.shape[3],
            ]
            pixel = subarray * weight[out_channel]
            pixel = np.array([np.sum(i) for i in pixel])
            output[:, out_channel, index[0], index[1]] = pixel
    # iterate over channel and add bias
    if bias is not None:
        for i in range(output.shape[1]):
            output[:, i, ...] += bias[i]
    return output


def linear(
    input: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    bias: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """
    Args:
        input: (N,in_features)
        weight: (out_features,in_features)
        bias: (out_features)
    """
    if input.ndim != 2:
        raise ValueError
    if input.shape[1] != weight.shape[1]:
        raise ValueError(
            "input features must be equal to weight in_features, "
            f"got input {input.shape} and weight {weight.shape}"
        )
    if bias is not None:
        if bias.ndim != 1:
            raise ValueError
        if bias.shape[0] != weight.shape[0]:
            raise ValueError
    output = np.zeros((input.shape[0], weight.shape[0]))
    for (index, in_feature) in enumerate(input):
        out_feature = weight @ in_feature.T
        if bias is not None:
            out_feature += bias
        output[index] = out_feature
    return output


def relu(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, x)


def _pool2d(
    x: npt.NDArray[np.float64],
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    pool_mode: Literal["max", "avg"] = "max",
):
    """
    This func do not handle the boundary separately (only accept integer
    as kernel_size and stride).
    """
    if x.ndim != 2:
        raise ValueError
    x = np.pad(x, padding, mode="constant")
    output_shape = (
        (x.shape[0] - kernel_size) // stride + 1,
        (x.shape[1] - kernel_size) // stride + 1,
    )
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (
        stride * x.strides[0],
        stride * x.strides[1],
        x.strides[0],
        x.strides[1],
    )
    A_w = as_strided(x, shape_w, strides_w)
    if pool_mode == "max":
        return A_w.max(axis=(2, 3))
    elif pool_mode == "avg":
        return A_w.mean(axis=(2, 3))


def pool2d(
    x: npt.NDArray[np.float64],
    kernel_size: int = 2,
    stride: int = 2,
    padding: int = 0,
    pool_mode: Literal["max", "avg"] = "max",
) -> npt.NDArray[np.float64]:
    """
    This function has no quality assurance, just duplicated.
    """
    if x.ndim != 4:
        raise ValueError
    output = np.zeros(
        (
            x.shape[0],
            x.shape[1],
            (x.shape[2] - kernel_size) // stride + 1,
            (x.shape[3] - kernel_size) // stride + 1,
        )
    )
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            output[i, j, ...] = _pool2d(
                x[i, j, ...],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pool_mode=pool_mode,
            )
    return output


def softmax(x):
    """useless in forward."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
