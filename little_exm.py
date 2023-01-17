import logging

# Create a logger
mainlogger = logging.getLogger("mainlogger")

# Change to logging.WARNING to disable logging statements
debug_level = logging.DEBUG
mainlogger.setLevel(debug_level)

# Create file handler
logfile = "output.log"
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(debug_level)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to logger
mainlogger.addHandler(fh)
mainlogger.addHandler(ch)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import transformer_badge as transformer
import model_train_construct as model_train
import copy_from_past as past
import scipy.io as scio
import os
from torchvision.datasets import MNIST

def greedy_show(model, src, src_mask, trg,size_cont,src_save):
    """
    Update src_save based on repeated greedy_decode().
    """
    mainlogger.info("Calling greedy_show()")
    # copy trg into CPU memory
    trg = trg.cpu()
    # repeat 10 times
    for ijk in range(10):
        mainlogger.debug("ijk = %s", ijk)
        # Set for_show to be the decoded version of model with start_symbol=1
        for_show = greedy_decode(ijk,model, src, src_mask, trg, start_symbol=1)
        mainlogger.debug("for_show = %s", for_show)
        # Set result to be for_show but as a 1 x (d0*d1) Tensor
        result = for_show.reshape([1,for_show.shape[0]*for_show.shape[1]])
        mainlogger.debug("result = %s", result)
        # Copy result into CPU memory
        result = result.cpu()
        mainlogger.debug("result copied to CPU")
        # Make g a 1x(size_cont^2) Tensor filled with 0s
        g = torch.zeros(size_cont*size_cont)
        mainlogger.debug("initial g = %s", g)
        mainlogger.debug("result.squeeze() - 1 = %s", result.squeeze() - 1)
        # First consider result with dimensions of length 1 removed, making it 1D
        # Set given element of g to be 1
        # URGENT: Do we want to set to 1?
        # TODO: examine dimensions
        g[result.squeeze() - 1] = 1
        mainlogger.debug("updated g = %s", g)
        # Reshape g to be size_cont x size_cont
        g = g.reshape(size_cont,size_cont)
        mainlogger.debug("reshaped g = %s", g)
        # Make b a 1x(size_cont^2) Tensor filled with 0s
        b = torch.zeros(size_cont*size_cont)
        mainlogger.debug("initial b = %s", b)
        mainlogger.debug("trg[ijk,:] - 1 = %s", trg[ijk,:] - 1)
        # Set given element of b to be 1
        # URGENT: Do we want to set to 1?
        # TODO: examine dimensions
        b[trg[ijk,:] - 1] = 1
        mainlogger.debug("updated b = %s", b)
        # Reshape b to be size_cont x size_cont
        mainlogger.debug(size_cont,size_cont)
        mainlogger.debug("reshaped b = %s", b)
        # Set first and last elements to 0
        b[0,0] = 0
        g[0,0] = 0
        b[-1,-1] = 0
        g[-1,-1] = 0
        mainlogger.debug("final b = %s", b)
        mainlogger.debug("final g = %s", g)
        # Convert b from Tensor into numpy array (process on CPU)
        b = b.numpy()
        # Convert g from Tensor into numpy array (process on CPU)
        g = g.numpy()
        # Set loss_raw to the sum of the elements of the absolute value of the difference between element ijk (loop iterator) of src_save and b
        loss_raw = abs(src_save[ijk]-b).sum()
        mainlogger.debug("loss_raw = %s", loss_raw)
        # Set loss to the sum of the elements of the absolute value of the difference between g and b 
        loss = abs(g-b).sum()
        mainlogger.debug("loss = %s", loss)
        # If loss < loss_raw, update src_save element ijk to be g
        if(loss<loss_raw):
            src_save[ijk] = g
    return src_save

def greedy_decode(ijk,model, src, src_mask, trg, start_symbol):
    """
    Take model and decode it.
    """
    mainlogger.info("Calling greedy_decode()")
    # Change Tensor src to be only the two rows given by ijk and ijk+1
    src = src[ijk:ijk+1, :]
    mainlogger.debug("src = %s", src)
    # Create Tensor of indices of nonzero values of the two rows of trg given by ijk and ijk+1
    # Set int max_length to be the length of the first dimension of this Tensor
    # Effectively, max_length = number of nonzero elements in the selected rows of trg
    max_length = trg[ijk:ijk+1,:].nonzero().shape[0]
    mainlogger.debug("max_length = %s", max_length)
    # Set src_mask to only be its first two rows and columns
    src_mask = src_mask[0:1, 0:1, :]
    mainlogger.debug("src_mask = %s", src_mask)
    # Set memory to be the encoded version of model using src and src_mask
    memory = model.encode(src, src_mask)
    mainlogger.debug("memory = %s", memory)
    # Save ys on current CUDA device (GPU) as a 1x1 Tensor containing start_symbol
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).cuda()
    mainlogger.debug("ys = %s", ys)
    # Repeat max_length - 1 times
    for i in range(max_length - 1):
        mainlogger.debug("i = %s", i)
        # Set out as decoded version of model
        out = model.decode(memory, src_mask, Variable(ys), Variable(transformer.subsequent_mask(ys.size(1)).type_as(src.data)))
        mainlogger.debug("out = %s", out)
        # Set prob as the generator of model with dimensions given by elements of matrix out
        prob = model.generator(out[:, -1])
        mainlogger.debug("prob = %s", prob)
        # Set throwaway variable and next_word (Tensors) as the maximum value element in dimension 1 of prob
        _, next_word = torch.max(prob, dim=1)
        # Make next_word into a scalar
        next_word = next_word.item()
        mainlogger.debug("next_word = %s", next_word)
        # Concatenate (append) in dimension 1 ys and a 1x1 Tensor containing next_word (which is on CUDA GPU device)
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(next_word).cuda()], dim=1)
        mainlogger.debug("ys = %s", ys)
    return ys

def trg_dealwith(input_image, imsize):
    """
    TODO: Determine effect of this function
    """
    mainlogger.info("Calling trg_dealwith()")
    # Set arrange_likeu to 1D Tensor containing integer values [1, imsize[0]^2 + 1]
    arrange_likeu = torch.arange(1, imsize[0] * imsize[0] + 1)
    mainlogger.debug("arrange_likeu = %s", arrange_likeu)
    # Reshape input_image so that second dimension (dimension 1) is imsize[0]^2 in length
    input_image = input_image.reshape(input_image.shape[0], imsize[0] * imsize[0])
    mainlogger.debug("reshaped input_image = %s", input_image)
    # Set trg to be element-wise product of input_image and arrange_likeus
    trg = input_image * arrange_likeu
    mainlogger.debug("trg = %s", trg)
    # Remove all dimensions that are length 1 from trg (make trg 1D?)
    trg = trg.squeeze()
    mainlogger.debug("squeezed trg = %s", trg)
    # Set find_max_dim to be the highest number of nonzero terms in any single row of trg
    find_max_dim = torch.count_nonzero(trg,dim=1).max()
    mainlogger.debug("find_max_dim = %s", find_max_dim)
    # Initialize trg_batch as a trg.shape[0] x find_max_dim Tensor filled with 0s
    trg_batch = torch.zeros(trg.shape[0],find_max_dim)
    mainlogger.debug("trg_batch = %s", trg_batch)
    # Initialize index_x as 0
    index_x = 0
    # Loop number of times equal to length of dimension 0 of trg
    while (index_x != trg.shape[0]):
        mainlogger.debug("index_x = %s", index_x)
        # Set trg_pice to row of trg given by index_x
        trg_pice = trg[index_x, :]
        mainlogger.debug("trg_pice = %s", trg_pice)
        # Set trg_nonzero to be Tensor of indices of nonzero elements of trg_pice
        trg_nonzero = trg_pice.nonzero()
        mainlogger.debug("trg_nonzero = %s", trg_nonzero)
        # Remove dimensions of length 1 from places where dimension 0 element is nonzero?
        trg_pice = trg_pice[trg_nonzero].squeeze()
        mainlogger.debug("trg_pice = %s", trg_pice)
        # Set row index_x of trg_batch to be trg_pice (up to length of trg_pice)
        trg_batch[index_x,0:trg_pice.shape[0]] = trg_pice
        mainlogger.debug("trg_batch = %s", trg_batch)
        # Increment index_x
        index_x += 1

    # Initialize trg_pice_zero as Tensor of 0s with same dimensions as trg_batch (dimension 1 length increased by 1)
    trg_pice_zero = torch.zeros(trg_batch.shape[0],trg_batch.shape[1]+1)
    mainlogger.debug("trg_pice_zero = %s", trg_pice_zero)
    # Set all but first column of trg_pice_zero to be equivalent to trg_batch
    trg_pice_zero[:,1:] = trg_batch
    mainlogger.debug("updated trg_pice_zero = %s", trg_pice_zero)
    # Copy trg_pice_zero to default CUDA device (GPU)
    trg_pice_zero = trg_pice_zero.cuda()
    return trg_pice_zero


def run_epoch(model,size_cont,readPatternFile,readImageFile,save_name,V2,src_save):
    """
    Standard Training and Logging Function
    """
    mainlogger.info("Calling run_epoch()")
    # Set image size to 32
    imsize = [32]
    mainlogger.info("imsize = %s", imsize)
    # Load readImageFile and save as input_image
    input_image = np.load(readImageFile)#change
    mainlogger.debug("input_image = %s", input_image)
    # Output string "input_image" and the shape stored in input_image
    mainlogger.info("input_image shape = %s",input_image.shape)
    # Load readPatternFile and save as pattern
    pattern = np.load(readPatternFile)
    mainlogger.debug("pattern = %s", pattern)
    # Set src_tender to be combination of input_image and pattern
    src_tender = src_dealwith(input_image, pattern,V2)
    mainlogger.debug("src_tender = %s", src_tender)
    # Convert input_image from numpy array to torch Tensor
    input_image = torch.from_numpy(input_image)
    #
    trg_tender = trg_dealwith(input_image,imsize)
    mainlogger.debug("trg_tender = %s", trg_tender)
    # Initialize a batch with src = src_tender, trg = trg_tender, and pad = 0
    batch = model_train.Batch(src_tender, trg_tender, 0)
    mainlogger.debug("batch = %s", batch)
    # Update src_save through repeated greedy_decode()
    src_save = greedy_show(model, batch.src, batch.src_mask, batch.trg, size_cont,src_save)
    mainlogger.debug("src_save = %s", src_save)
    # Save src_save to binary file in .npy format
    np.save(save_name,src_save)#change
    return src_save


def src_dealwith(img_ori, pattern,V2):
    """
    Combine image with speckle pattern.
    """
    mainlogger.info("Calling src_dealwith()")
    # Remove dimensions of length 1 from pattern
    pattern = pattern.squeeze()
    mainlogger.debug("squeeded pattern = %s", pattern)
    # Reshape pattern to be 1 x (original dimension 0 length) x 32 x 32
    # NOTE: This may introduce expensive runtimes based on input size
    pattern = pattern.reshape(1, pattern.shape[0], 32, 32)
    mainlogger.debug("reshaped pattern = %s", pattern)
    # Reshape img_ori to be 10 x 1 x 32 x 32
    img_ori = img_ori.reshape(10, 1, 32, 32)
    mainlogger.debug("reshaped img_ori = %s", img_ori)
    # Create image from element-wise multiplication of pattern and img_ori (original image)
    # URGENT: divide by 255?
    image = pattern * img_ori
    mainlogger.debug("image = %s", image)
    # Convert image from numpy array to torch Tensor
    image = torch.from_numpy(image)
    # Set I to be 32 x 32 Tensor, result of dimensions 2 and 3 summed over in image
    I = torch.sum(image, (2, 3))
    mainlogger.debug("initial I = %s", I)
    # Copy I to default CUDA device (GPU)
    I = I.cuda()
    # Set I_min and I_index to have the minimum value in dimension 1 of I
    I_min, I_index = torch.min(I,1)
    mainlogger.debug("I_min = %s", I_min)
    # Reshape I_min to have original length in dimension 0 and length 1 in dimension 1
    I_min = I_min.reshape(I.shape[0],1)
    mainlogger.debug("reshaped I_min = %s", I_min)
    # Set I to be difference between I and I_min
    I = I - I_min
    mainlogger.debug("I = %s", I)
    # Set I_max and I_index to be maximum value in dimension 1 of I
    I_max, I_index = torch.max(I,1)
    mainlogger.debug("I_max = %s", I_max)
    # Reshape I_max to have original length in dimension 0 and length 1 in dimension 1
    I_max = I_max.reshape(I.shape[0],1)
    mainlogger.debug("reshaped I_max = %s", I_max)
    # Manually set which operation is done, whether noise is introduced
    if(0):
        # Update I
        I = I / (I_max + 1) * (V2 * 0.95)
        # Generate randy Tensor of random values with same shape as I; values in range [-0.025, 0.025)
        # Copy to current CUDA device (GPU)
        randy = ((torch.rand(I.shape[0],I.shape[1])-0.5)*0.05).cuda()
        # Introduce randy to I to make I_noise
        I_noise = I*randy
        # Add I_noise to I
        I = I+I_noise
    else:
        # Update I
        I = I/(I_max+1)*V2
        mainlogger.debug("I = %s", I)
    # Cast elements of I to int, rounding down
    # URGENT: This may be where rounding takes place
    I = I.int()
    mainlogger.debug("rounded I = %s", I)
    return I



# !!!!!!!!!!!ray15 nuber 改900
#!!!!!!!!!!!!!!!!检查是不是V2
# readImageFile = "/scratch/user/taopeng/REAL_TRUE_white/A_WHR_MAKE_RESULT/image/Number_Image.npy"
readImageFile = "./image/Smile_image.npy"

# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR2p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR3p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR5p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Ray_SR5p.npy"
readPatternFile = "./pattern/pink_p5.npy"

# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR2p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR3p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR5p_5/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Ray_SR5p_10/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Ray_SR15p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_PINK_SR2p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_PINK_SR3p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Pink_SR5p_10/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Ray_SR5p_10/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Ray_SR15p_0/Modelpara.pth"
# readModelFile = "./models/test_Modelpara.pth"

saveModelFile = './zmodel/grayscale_model_alpha'

# save_name = './result/Noise_Pink_SR2p_0.npy'
# save_name = './result/Noise_Pink_SR3p_0.npy'
# save_name = './result/Noise_Pink_SR5p_5.npy'
# save_name = './result/Noise_Ray_SR5p_10.npy'
# save_name = './result/Noise_Ray_SR15p_0.npy'
# save_name = './result/SMILE_PINK_SR2p_0.npy'
# save_name = './result/SMILE_PINK_SR3p_0.npy'
# save_name = './result/SMILE_Pink_SR5p_10.npy'
# save_name = './result/SMILE_Ray_SR5p_10.npy'
save_name = './zresult/SMILE_Pink_p5_grayscale_1.npy'



size_cont = 32
V1 = size_cont * size_cont + 1
V2 = size_cont * size_cont + 1

# TODO: Learn what these do
DataLoaderName = MNIST
batch = 10 #200
imsize =[size_cont]
criterion = nn.CrossEntropyLoss()
# Construct blank model with structure
model = transformer.make_model(V1, V2,N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
# Read model file if there is existing one
# model.load_state_dict(torch.load(readModelFile))#change
model = model.cuda()
model_opt = model_train.NoamOpt(model.src_embed[0].d_model, 1, 400,
                                torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9))
# src_save = np.load(os.path.join(SaveModelFile, 'lab_trg_32_JUly20.npy'))
src_save = np.ones([10,32,32])*900
for epoch in range(1000):
    model.train()
    # model.eval()
    print("Epoch: ", epoch + 1)
    start = time.time()
    src_save = run_epoch(model,size_cont,readPatternFile,readImageFile,save_name,V2,src_save)
    torch.save(model.state_dict(), saveModelFile)
