import pickle

import numpy as np


with open("dump/keras_flatten.pkl", "rb") as f:
    framework_result = pickle.load(f)

with open("dump/numpy_flatten.pkl", "rb") as f:
    pure_result = pickle.load(f)


print("test flattened result")
print(np.allclose(framework_result, pure_result, rtol=0.000001))
print(np.allclose(framework_result, pure_result, rtol=0.0001))
