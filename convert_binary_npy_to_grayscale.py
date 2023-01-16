import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from copy_from_past import *

smile = np.load('./image/Smile_image.npy')

counter = 255

(x, y, z) = np.shape(smile)

for i in range(x):
    for j in range(y):
        for k in range(z):
            if smile[i, j, k] == 1:
                if counter <= 150:
                    counter = 255
                smile[i, j, k] = counter
                counter = counter - 1

np.save('./image/Smile_image_grayscale.npy', smile)