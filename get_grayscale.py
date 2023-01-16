import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from copy_from_past import *


# change this to 3 to upgrade grayscale images to RGB size
num_output_channels = 1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/mnt/c/Users/zachb/research/deep_learning/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToPILImage(),
                                    torchvision.transforms.Grayscale(num_output_channels),
                                    torchvision.transforms.Pad(4)])),
    batch_size=10)

# Load and convert to nparray
# save nparray as npy files