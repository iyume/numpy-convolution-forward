"""
Convolutional Neural Network components implemented in pure numpy.
"""
from typing import Any, Callable, Iterable, Iterator


class Module:
    # trick mypy
    forward: Callable[..., Any] = lambda: ...

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, inputs: Iterable[Module]) -> None:
        self._modules = list(inputs)

    def forward(self, x):
        for m in self._modules:
            x = m(x)
        return x

    def add_module(self, module: Module) -> None:
        self._modules.append(module)

    def __getitem__(self, key: int) -> Module:
        return self._modules[key]

    def __iter__(self) -> Iterator:
        return iter(self._modules)
