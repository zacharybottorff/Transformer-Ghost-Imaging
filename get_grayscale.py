import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision


# change this to 3 to upgrade grayscale images to RGB size (still gray though)
num_output_channels = 1

batch_size_custom = 10
# Get set of MNIST images for training
mnist_trainset = torchvision.datasets.MNIST(root='./files', train=True, download=True,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Grayscale(num_output_channels),
                                                torchvision.transforms.Pad(4)])),
# Get different set of MNIST images for testing
mnist_testset = torchvision.datasets.MNIST(root='./files', train=False, download=True,
                                           transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Grayscale(num_output_channels),
                                                torchvision.transforms.Pad(4)])),
# Make training DataLoader
train_loader = data.DataLoader(mnist_trainset, batch_size=batch_size_custom, shuffle=True)
# Make testing DataLoader
test_loader = data.DataLoader(mnist_testset, batch_size=batch_size_custom, shuffle=True)

# Load and convert to nparray
# save nparray as npy files