import pickle

import numpy as np


with open("dump/keras_flatten.pkl", "rb") as f:
    framework_result = pickle.load(f)

with open("dump/numpy_flatten.pkl", "rb") as f:
    pure_result = pickle.load(f)


print("test flattened result")
print(np.allclose(framework_result, pure_result, rtol=1e-6))
print(np.allclose(framework_result, pure_result, rtol=1e-5))
print(np.allclose(framework_result, pure_result, rtol=1e-4))
