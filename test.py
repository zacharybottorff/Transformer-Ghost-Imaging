import torch
import numpy as np

x = np.array([1, 0, 1, 0, 2])

x = torch.from_numpy(x)

pad = 0

print("x = ", x)
print("x != pad = ", x != pad)
print("(x != pad).unsqueeze(-2) = ", (x != pad).unsqueeze(-2))

class Fun(object):
    def __init__(self, x):
        self.y = x

    def __call__(self, x):
        x += 1
        return x

ha = Fun(10)

print("ha = ", ha)
print("ha.y = ", ha.y)
print("ha(12) = ", ha(12))