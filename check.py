import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Array of images
a = np.load('./result/SMILE_Pink_SR5p_10.npy') #32,32

# repeat for 10 images
for i in range(10):
  # put image i in figure
  # TODO: consider different normalization modes of imshow()
  # NOTE: may need to convert vmax to 1
  plt.imshow(a[i], cmap='gray', vmin=0, vmax=255)
  # turn off axis
  plt.axis('off')
  # display open figure
  plt.show()
