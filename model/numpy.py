from typing import Dict, Union
import numpy as np

from nn import functional as F, Module, Sequential, Conv2d, Linear, ReLU


class VGG19pure(Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = Sequential(
            (
                Conv2d(3, 64, 3, padding=1),
                ReLU(),
                Conv2d(64, 64, 3, padding=1),
                ReLU(),
            )
        )
        self.block2 = Sequential(
            (
                Conv2d(64, 128, 3, padding=1),
                ReLU(),
                Conv2d(128, 128, 3, padding=1),
                ReLU(),
            )
        )
        self.block3 = Sequential(
            (
                Conv2d(128, 256, 3, padding=1),
                ReLU(),
                Conv2d(256, 256, 3, padding=1),
                ReLU(),
                Conv2d(256, 256, 3, padding=1),
                ReLU(),
                Conv2d(256, 256, 3, padding=1),
                ReLU(),
            )
        )
        self.block4 = Sequential(
            (
                Conv2d(256, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
            )
        )
        self.block5 = Sequential(
            (
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
                Conv2d(512, 512, 3, padding=1),
                ReLU(),
            )
        )
        self.fc1 = Linear(512, 512)
        self.fc2 = Linear(512, 512)
        self.aux = Linear(512, 10)
        self.load_weights()

    @staticmethod
    def _set_weight(parameter, weight, bias):
        if parameter.weight.shape != weight.shape or parameter.bias.shape != bias.shape:
            raise RuntimeError
        parameter.weight = weight
        parameter.bias = bias

    def load_weights(self) -> None:
        import pickle

        with open("dump/epoch30.pkl", "rb") as f:
            dct: Dict[str, np.ndarray] = pickle.load(f)

        parameter: Union[Conv2d, Linear]
        # load conv2d weight and bias
        num_conv = (2, 2, 4, 4, 4)
        for i in range(5):
            for j in range(num_conv[i]):
                conv_name = f"block{i+1}_conv{j+1}"
                weight = dct[f"{conv_name}/kernel:0"]
                bias = dct[f"{conv_name}/bias:0"]
                weight = weight.transpose(3, 2, 0, 1)
                conv_lst = [
                    i for i in getattr(self, f"block{i+1}") if isinstance(i, Conv2d)
                ]
                parameter = conv_lst[j]
                self._set_weight(parameter, weight, bias)
        weight = dct["dense/kernel:0"]
        bias = dct["dense/bias:0"]
        weight = weight.T
        parameter = self.fc1
        self._set_weight(parameter, weight, bias)
        weight = dct["dense_1/kernel:0"]
        bias = dct["dense_1/bias:0"]
        weight = weight.T
        parameter = self.fc2
        self._set_weight(parameter, weight, bias)
        weight = dct["dense_2/kernel:0"]
        bias = dct["dense_2/bias:0"]
        weight = weight.T
        parameter = self.aux
        self._set_weight(parameter, weight, bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError

        from time import time

        print("block1 forwarding...", end="")
        st = time()
        x = self.block1(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 150s

        x = F.pool2d(x)

        print("block2 forwarding...", end="")
        st = time()
        x = self.block2(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 110s

        x = F.pool2d(x)

        print("block3 forwarding...", end="")
        st = time()
        x = self.block3(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 250s

        x = F.pool2d(x)

        print("block4 forwarding...", end="")
        st = time()
        x = self.block4(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 300s

        x = F.pool2d(x)

        print("block5 forwarding...", end="")
        st = time()
        x = self.block5(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 120s

        x = F.pool2d(x)

        x = x.reshape(x.shape[0], -1)

        # import pickle

        # with open("dump/numpy_flatten.pkl", "wb") as f:
        #     pickle.dump(x, f)
        # print('dumped to "dump/numpy_flatten.pkl"')

        # import pickle

        # with open("dump/numpy_flatten.pkl", "rb") as f:
        #     x = pickle.load(f)
        # print('recovered from "dump/numpy_flatten.pkl"')

        print("fc1 forwarding...", end="")
        st = time()
        x = self.fc1(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 0.2s

        x = F.relu(x)

        print("fc2 forwarding...", end="")
        st = time()
        x = self.fc2(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 0.2s

        x = F.relu(x)

        print("aux forwarding...", end="")
        st = time()
        x = self.aux(x)
        et = time()
        print(f"{et-st:.4f}s")
        # about 0.001s

        x = F.softmax(x)

        return x
