# Numpy Convolution Forward

Numpy only convolutional neural network forward implementation, implement the Conv2d, Linear, Pool2d, Activation, etc.

## Accuracy

In my test case:

```
epoch: 30
num_test: 100
```

Accuracy result:

```
keras: 0.78
numpy: 0.70
```

The result of `test_units.py` True at `rtol=1e-4`, which means the flattened result (before linear) is very close.
