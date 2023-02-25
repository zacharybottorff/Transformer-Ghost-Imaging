# Developed by Zachary D. Bottorff as part of a thesis
# April 2023
# Based on the Transformer Network by
    # Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    # Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin
# And on the Ghost Translation regime by
    # Wenhan Ren, Xiaoyu Nie, Tao Peng, and Marlan O. Scully

import numpy as np
import torch
import torch.nn as nn
import transformer_badge as transformer
import model_train_construct as model_train
import copy_from_past as past
import torchvision.datasets as tvdata



# Set files to be used
readImageFile = "./image/Smile_image_grayscale.npy"
readPatternFile = "./pattern/pink_p5.npy"
readModelFile = "./zmodel/grayscale_model_beta.pth"
saveModelFile = "./zmodel/grayscale_model_beta.pth"
saveName = "./zresult/SMILE_Pink_p5_grayscale_pe.npy"


# Model parameters
# Set image size
size_cont = 32
# Set model dimension number
d_model = 512
# Set feed forward dimension number
d_ff = 2048
# Set number of layers (note d_k = d_model / h)
# Where d_k is dimension of query/key/value vectors
layers = 8
# Set size of source vocabulary, a dictionary of embeddings related to src
V_src = 256
# Set size of target vocabulary, a dictionary of embeddings related to trg
V_trg = 256
# Set training mode
mode = 'train' # 'eval'

# Set data loader
Data_loader_name = tvdata.MNIST
# Set batch size
batch = 10
# Set criterion for computing loss
loss_criterion = nn.CrossEntropyLoss()



