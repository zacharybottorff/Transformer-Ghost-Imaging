import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Array of images
a = np.load('./zresult/SMILE_Pink_p5_10000.npy') #32,32

# Original images
b = np.load('./image/Smile_image.npy')

# repeat for 10 images
for i in range(10):
  # Make blank figure
  fig = plt.figure(figsize=(16, 8))
  rows = 1
  columns = 2
  fig.add_subplot(rows, columns, 1)
  # put image i in figure
  # TODO: consider different normalization modes of imshow()
  # NOTE: may need to convert vmax to 1
  plt.imshow(b[i], cmap='gray', vmin=0, vmax=1)
  # turn off axis
  plt.axis('off')
  
  plt.title('Input')
  
  fig.add_subplot(rows, columns, 2)
  
  plt.imshow(a[i], cmap='gray', vmin=0, vmax=1)
  
  plt.axis('off')
  
  plt.title('Output')
  
  # display open figure
  plt.show()
