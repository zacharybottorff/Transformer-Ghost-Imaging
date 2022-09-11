import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from transformer_badge import *
from model_train_construct import *
from copy_from_past import *
import scipy.io as scio
import os

def greedy_show(model, src, src_mask, trg,size_cont,src_save):
  # repeat 10 times
  for ijk in range(10):
    for_show = greedy_decode(ijk,model, src, src_mask, trg, start_symbol=1)
    result = for_show.reshape([1,for_show.shape[0]*for_show.shape[1] ])
    result = result.cpu()
    trg = trg.cpu()
    g = torch.zeros(size_cont*size_cont)
    g[result.squeeze () - 1] = 1
    g = g.reshape(size_cont,size_cont)
    b = torch.zeros(size_cont*size_cont)
    b[(trg[ijk,:] - 1)] = 1
    b = b.reshape(size_cont,size_cont)
    b[0,0] = 0
    g[0,0] = 0
    b[31,31] = 0
    g[31,31] = 0
    b = b.numpy()
    g = g.numpy()
    loss_raw = abs(src_save[ijk]-b).sum()
    loss = abs(g-b).sum()
    if(loss<loss_raw):
      src_save[ijk] = g
  return src_save




def greedy_decode(ijk,model, src, src_mask, trg, start_symbol):
    src = src[ijk:ijk+1, :]
    max_length = trg[ijk:ijk+1,:].nonzero().shape[0]
    src_mask = src_mask[0:1, 0:1, :]
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).cuda()
    # repeat max_length - 1 times
    for i in range(max_length - 1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        # make next_word refer to its item
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(next_word).cuda()], dim=1)
    return ys

def trg_dealwith(input_image, imsize):
    arrange_likeu = torch.arange(1, imsize[0] * imsize[0] + 1)
    input_image = input_image.reshape(input_image.shape[0], imsize[0] * imsize[0])
    trg = input_image * arrange_likeu
    trg = trg.squeeze()
    # print("Line 420,trg.shape ",trg.shape) [32,900]
    find_max_dim = torch.count_nonzero(trg,dim=1).max()
    trg_batch = torch.zeros(trg.shape[0],find_max_dim)
    index_x = 0
    while (index_x != trg.shape[0]):
        trg_pice = trg[index_x, :]
        trg_nonzero = trg_pice.nonzero()
        trg_pice = trg_pice[trg_nonzero].squeeze()
        trg_batch[index_x,0:trg_pice.shape[0]] = trg_pice
        index_x += 1
   
    trg_pice_zero = torch.zeros(trg_batch.shape[0],trg_batch.shape[1]+1)
    trg_pice_zero[:,1:] = trg_batch
    trg_pice_zero = trg_pice_zero.cuda()
    # print("Line 434,trg_pice_zero.shape ",trg_pice_zero.shape)
    return trg_pice_zero


def run_epoch(model,size_cont,readPatternFile,readImageFile,save_name,V2,src_save):
    """
    Standard Training and Logging Function
    """
    # set image size to 32
    imsize = [32]
    # load readImageFile and save as input_image
    input_image = np.load(readImageFile)#change
    # output string "input_image" and the shape stored in input_image
    print("input_image",input_image.shape)
    # load readPatternFile and save as pattern
    pattern = np.load(readPatternFile)
    #
    src_tender = src_dealwith(input_image, pattern,V2)
    #
    input_image = torch.from_numpy(input_image)
    #
    trg_tender = trg_dealwith(input_image,imsize)
    # Initialize a batch with src = src_tender, trg = trg_tender, and pad = 0
    batch = Batch(src_tender, trg_tender, 0)
    #
    src_save = greedy_show(model, batch.src, batch.src_mask, batch.trg, size_cont,src_save)
    #
    np.save(save_name,src_save)#change
    #
    return src_save


def src_dealwith(img_ori, pattern,V2):
    pattern = pattern.squeeze()
    pattern = pattern.reshape(1, pattern.shape[0], 32, 32)
    img_ori = img_ori.reshape(10, 1, 32, 32)
    image = pattern * img_ori
    image = torch.from_numpy(image)
    I = torch.sum(image, (2, 3))
    I = I.cuda()
    I_min,I_index = torch.min(I,1)
    I_min =I_min.reshape(I.shape[0],1)
    I = I - I_min
    I_max,I_index = torch.max(I,1)
    I_max =I_max.reshape(I.shape[0],1)
    if(0):
        I = I / (I_max + 1) * (V2 * 0.95)
        randy = ((torch.rand(I.shape[0],I.shape[1])-0.5)*0.05).cuda()
        I_noise = I*randy
        I = I+I_noise
    else:
        I = I/(I_max+1)*V2
    I = I.int()
    return I



    return I


# !!!!!!!!!!!ray15 nuber 改900
#!!!!!!!!!!!!!!!!检查是不是V2
# readImageFile = "/scratch/user/taopeng/REAL_TRUE_white/A_WHR_MAKE_RESULT/image/Number_Image.npy"
readImageFile = "/scratch/user/taopeng/REAL_TRUE_white/A_WHR_MAKE_RESULT/image/Smile_image.npy"

# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR2p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR3p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Pink_SR5p.npy"
# readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Ray_SR5p.npy"
readPatternFile = "/scratch/user/taopeng/REAL_TRUE_white/data/pattern/Ray_SR15p.npy"

# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR2p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR3p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Pink_SR5p_5/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Ray_SR5p_10/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/Noise_Ray_SR15p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_PINK_SR2p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_PINK_SR3p_0/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Pink_SR5p_10/Modelpara.pth"
# readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Ray_SR5p_10/Modelpara.pth"
readModelFile = "/scratch/user/taopeng/REAL_TRUE_white/data/SMILE_Ray_SR15p_0/Modelpara.pth"


# save_name = './result/Noise_Pink_SR2p_0.npy'
# save_name = './result/Noise_Pink_SR3p_0.npy'
# save_name = './result/Noise_Pink_SR5p_5.npy'
# save_name = './result/Noise_Ray_SR5p_10.npy'
# save_name = './result/Noise_Ray_SR15p_0.npy'
# save_name = './result/SMILE_PINK_SR2p_0.npy'
# save_name = './result/SMILE_PINK_SR3p_0.npy'
# save_name = './result/SMILE_Pink_SR5p_10.npy'
# save_name = './result/SMILE_Ray_SR5p_10.npy'
save_name = './result/SMILE_Ray_SR15p_0.npy'



size_cont = 32
V1 = size_cont * size_cont + 1
V2 = size_cont * size_cont + 1


DataLoaderName = MNIST
batch = 10 #200
imsize =[size_cont]
criterion = nn.CrossEntropyLoss()
model = make_model(V1, V2,N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
model.load_state_dict(torch.load(readModelFile))#change
model = model.cuda()
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9))
# src_save = np.load(os.path.join(SaveModelFile, 'lab_trg_32_JUly20.npy'))
src_save = np.ones([10,32,32])*900
for epoch in range(1000):
    model.train()
    start = time.time()
    src_save = run_epoch(model,size_cont,readPatternFile,readImageFile,save_name,V2,src_save)
 
