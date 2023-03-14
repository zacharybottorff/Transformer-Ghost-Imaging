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
# import copy_from_past as past
import time
import torchvision.datasets as tvdata

def info(variable_name, variable, shape=False):
    """
    Shorthand for using the logger.info() function.
    Output is in the form variable_name = variable
    """
    if shape:
        transformer.mainlogger.info("%s%s%s" % (variable_name, ".shape = ", variable.shape))
        transformer.mainlogger.info("%s%s%s" % (variable_name, " = ", variable))
    else:
        transformer.mainlogger.info("%s%s%s" % (variable_name, " = ", variable))
        
def debug(variable_name, variable, shape=False):
    """
    Shorthand for using the logger.debug() function.
    Output is in the form variable_name = variable
    """
    if shape:
        transformer.mainlogger.debug("%s%s%s" % (variable_name, ".shape = ", variable.shape))
        transformer.mainlogger.debug("%s%s%s" % (variable_name, " = ", variable))
    else:
        transformer.mainlogger.debug("%s%s%s" % (variable_name, " = ", variable))



def src_dealwith(img_ori, pattern, V_src, size_cont):
    """
    Combine image with speckle pattern.
    """
    transformer.mainlogger.info("Calling src_dealwith()")
    # Remove dimensions of length 1 from pattern
    pattern = pattern.squeeze()
    debug("pattern", pattern, shape=True)
    # Reshape pattern to be 1 x (original dimension 0 length) x 32 x 32
    pattern = pattern.reshape(1, pattern.shape[0], 32, 32)
    debug("reshaped pattern", pattern, shape=True)
    # Reshape img_ori to be 10 x 1 x 32 x 32
    img_ori = img_ori.reshape(10, 1, size_cont, size_cont)
    debug("reshaped img_ori", img_ori)
    # Create image from element-wise multiplication of pattern and img_ori (original image)
    image = pattern * img_ori
    debug("image", image, shape=True)
    # Convert image from numpy array to torch Tensor
    image = torch.from_numpy(image)
    # Set I to be result of dimensions 2 and 3 summed over in image
    I = torch.sum(image, (2, 3))
    debug("initial I", I, shape=True)
    # Copy I to default CUDA device (GPU)
    I = I.cuda()
    # Set I_min to have the minimum value in each unit of dimension 1 of I
    # Set I_index to be indices of minimum values
    I_min, I_index = torch.min(I, 1)
    debug("I_min", I_min)
    # Reshape I_min to have original length in dimension 0 and length 1 in dimension 1
    I_min = I_min.reshape(I.shape[0], 1)
    debug("reshaped I_min", I_min, shape=True)
    # Let I_min be our new zero point
    I = I - I_min
    debug("I", I, shape=True)
    # Set I_max to be maximum values in each unit of dimension 1 of I
    # Set I_index to be indices of maximum values
    I_max, I_index = torch.max(I, 1)
    debug("I_max", I_max, shape=True)
    # Reshape I_max to have original length in dimension 0 and length 1 in dimension 1
    I_max = I_max.reshape(I.shape[0], 1)
    debug("reshaped I_max", I_max, shape=True)
    # Set I on a relative scale from 0 to 255 * 32 * 32 (in other words, V_src minus 1*32*32)
    # NOTE: I_max was substituted for (I_max+1)
    I = I / I_max * (V_src - size_cont*size_cont)
    debug("I", I, shape=True)
    # Round to integer. I is 1D bucket detector signals
    I = I.int()
    debug("rounded I", I, shape=True)
    return I

def trg_dealwith(input_image, size_cont, batch_size):
    """
    Determine trg based on input_image.
    """
    transformer.mainlogger.info("Calling trg_dealwith()")
    debug("initial input_image", input_image, shape=True)
    # Reshape input_image to be (batch size) set of 1D Tensors
    input_image = input_image.reshape(batch_size, size_cont*size_cont)
    debug("reshaped input_image", input_image, shape=True)
    # Copy to GPU
    trg_tender = input_image.cuda()
    return trg_tender

def run_epoch(model, size_cont, pattern, input_image, saveName, V_src, in_progress, batch_size, loss, batch):
    """
    Standard Training and Logging Function
    """
    transformer.mainlogger.info("Calling run_epoch()")
    start = time.time()
    in_progress = greedy_show(model, batch.src, batch.src_mask, batch.trg, size_cont, in_progress, batch_size, loss, V_src)
    debug("in_progress", in_progress)
    # Save in_progress to file in .npy format
    np.save(saveName, in_progress)
    return in_progress

def greedy_show(model, src, src_mask, trg, size_cont, in_progress, batch_size, loss, V_src):
    """
    Update in_progress based on repeated greedy_decode()
    """
    transformer.mainlogger.info("Calling greedy_show()")
    # copy trg into CPU
    trg = trg.cpu()
    # repeat batch_size times
    for i in range(batch_size):
        debug("i", i)
        # NOTE: This cheats a little bit by takin the first pixel
        start_symbol = trg[i, 0]
        debug("trg", trg, shape=True)
        debug("start_symbol", start_symbol)
        # Set ys to be the decoded version of model with start symbol
        ys, memory = greedy_decode(i, model, src, src_mask, trg, start_symbol, loss)
        debug("ys", ys, shape=True)
        # Set result to be ys with correct dimensions
        result = ys.reshape([1,ys.shape[0]*ys.shape[1]])
        # Copy result to CPU, convert to numpy array
        result = result.cpu()
        result = result.numpy()
        debug("result", result, shape=True)
        
        # Set numpy version of trg to be trg_num
        trg_num = trg.numpy()
        debug("trg_num", trg_num, shape=True)
        # Set loss (total difference between image pixel values) from most recent epoch to loss_raw
        loss_old = abs(in_progress[i] - trg_num[i]).sum()
        debug("loss_old", loss_old, shape=True)
        # Set loss (total difference between image pixel values) from current epoch to be loss_new
        loss_new = abs(in_progress[i] - result).sum()
        debug("loss_new", loss_new, shape=True)
        # Set the image with the lower loss to be the updated in_progress
        if loss_new < loss_old:
            in_progress[i] = result
            transformer.mainlogger.debug("Updating in_progress.")
        
        # # Set current image of trg to be trg_im
        # trg_im = trg[i]
        # # Convert trg_im to numpy array
        # trg_im = trg_im.numpy()
        # # Make Tensor version of in_progress and copy to GPU
        # progress = torch.from_numpy(in_progress).cuda()
        # # See if loss has been reduced
        # # loss_raw = abs(in_progress[i] - trg_im).sum()
        # # Format this element of progress for loss computation
        # prog = torch.zeros(trg.shape[1], V_src).cuda()
        # # URGENT: THIS IS WRONG: SEE https://pytorch.org/docs/0.3.0/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
        # # prog = progress[i].contiguous().view(-1, progress[i].size(-1)).cuda()
        # debug("prog", prog, shape=True)
        # # Set 1D version of ys to be ys_1d
        # ys_1d = ys.squeeze()
        # # Let elements of prog be the guesses for each pixel (denoted by unit vector in dimension corresponding to value). 
        # # Because of greedy decoding, probability distributions are only 1 or 0
        # for k in range(prog.shape[0]):
        #     value = ys_1d[k].item()
        #     prog[k,value] = 1
        # debug("prog", prog, shape=True)
        # # Format this element of trg for loss computation
        # debug("trg[i]", trg[i], shape=True)
        # trgi = trg[i].long().contiguous().cuda()
        # debug("trgi", trgi, shape=True)
        # loss_raw = loss(prog, trgi).sum().backward().item() #### ABANDON CROSS-ENTROPY LOSS FOR NOW
        # debug("loss_raw", loss_raw, shape=True)
        # new_loss = loss(result.contiguous().view(-1, result.size(-1)),
        #                 trgi).sum().backward().item()
        # debug("new_loss", new_loss, shape=True)
        # if torch.abs(new_loss) < torch.abs(loss_raw): #####################
        #     # If so, update in_progress
        #     in_progress[i] = result
        #     transformer.mainlogger.debug("Updating in_progress.")
        
        
    return in_progress

def greedy_decode(i, model, src, src_mask, trg, start_symbol, loss):
    """
    Take model and decode it.
    """
    transformer.mainlogger.info("Calling greedy_decode()")
    # Only use the image out of batch given by i without reducing number of dimensions
    src = src[i:i+1, :]
    debug("src", src, shape=True)
    # Update src_mask
    # URGENT: I think the old way of doing this was totally wrong
    debug("src_mask", src_mask, shape=True)
    src_mask = src_mask[i:i+1, :]
    debug("updated src_mask", src_mask, shape=True)
    # Set memory to be the encoded version of model
    memory = model.encode(src, src_mask)
    debug("memory", memory)
    # Save ys to GPU as a 1 x 1 Tensor containing start_symbol
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).type_as(src.data).cuda()
    debug("ys", ys, shape=True)
    # Repeat max length - 1 times
    for j in range(size_cont*size_cont-1):
        debug("j", j)
        # Set out to be decoded version of model
        out = model.decode(memory, src_mask, model_train.Variable(ys), 
                           model_train.Variable(transformer.subsequent_mask(ys.size(1)).type_as(src.data)))
        debug("out", out, shape=True)
        # Set prob as the generator of model with dimensions given by elements of matrix out
        # This is a Tensor of probabilities for each value
        prob = model.generator(out[:, -1])
        debug("prob", prob, shape=True)
        # Set throwaway variable to be Tensor of maximum values in dimension 1
        # Set next_word to be Tensor of indices of those values
        # This identifies the most probable value in the src vector, i.e., most likely grayscale value
        _, next_word = torch.max(prob, dim=1)
        debug("next_word", next_word, shape=True)
        # Make next_word into a scalar
        next_word = next_word.item()
        debug("scalar next_word", next_word)
        # Create a 1 x 1 Tensor containing only next word on CUDA GPU device
        # Concatenate (append) next_word Tensor to ys in row dimension
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(next_word).cuda()], dim=1)
        debug("ys", ys, shape=True)
    return ys, memory
        


# Set files to be used
readImageFile = "./image/Smile_image_grayscale.npy"
readPatternFile = "./pattern/pink_p5.npy"
readModelFile = "./zmodel/grayscale_model_beta.pth"
saveModelFile = "./zmodel/grayscale_model_beta.pth"
saveName = "./zresult/SMILE_Pink_p5_grayscale_pe.npy"


# Model parameters
# Set image size
size_cont = 32
info("size_cont", size_cont)
# Set model dimension number
dim_model = 512
info("dim_model", dim_model)
# Set feed forward dimension number
dim_ff = 2048
info("dim_ff", dim_ff)
# Set number of layers (note d_k = d_model / h)
# Where d_k is dimension of query/key/value vectors
layers = 8
info("layers", layers)
# Set size of target vocabulary, a dictionary of embeddings related to trg
V_trg = 256
info("V_trg", V_trg)
# Set size of source vocabulary, a dictionary of embeddings related to src
V_src = V_trg * size_cont * size_cont
info("V_src", V_src)
# Set training mode
mode = "train" # "eval"
info("training mode", mode)
# Set whether continuing from saved model
old_model = False
info("old_model", old_model)

# Set data loader
Data_loader_name = tvdata.MNIST
# Set batch size
batch_size = 10
info("batch_size", batch_size)
# Set criterion for computing loss
loss_criterion = nn.CrossEntropyLoss()
# Construct a blank model with structure
model = transformer.make_model(V_src, V_trg, N=6, d_model=dim_model,
                               d_ff=dim_ff, h=layers, dropout=0.1)
# Construct optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, 
                             betas=(0.9, 0.98), eps=1e-9)

if not old_model:
    # # Set model parameters to be float
    # model.float()
    # Move model to GPU
    model.cuda()
    model_opt = model_train.NoamOpt(model.src_embed[0].d_model, 1, 400, optimizer)
    # Initialize in_progress with same number of pixels as input, but all pixels are 900
    in_progress = np.ones([batch_size,size_cont*size_cont])*900
    debug("in_progress", in_progress, shape=True)
    if mode == "train":
        # Set training mode
        model.train()
    
    # Load images from file
    input_image = np.load(readImageFile)
    debug("input_image", input_image, shape=True)
    # Load patterns from file
    pattern = np.load(readPatternFile)
    debug("pattern", pattern, shape=True)
    # Set initial src value
    src_tender = src_dealwith(input_image, pattern, V_src, size_cont)
    debug("src_tender", src_tender, shape=True)
    # Convert input_image from numpy array to torch Tensor
    input_image = torch.from_numpy(input_image)
    # Set initial trg value
    trg_tender = trg_dealwith(input_image, size_cont, batch_size)
    debug("trg_tender", trg_tender, shape=True)
    # Add extra 0 at end of trg (the Batch removes the last entry of dimension 1)
    trg_tender = torch.cat([trg_tender, torch.ones(10, 1, dtype=torch.long).fill_(0).cuda()], dim=1)
    debug("expanded trg_tender", trg_tender, shape=True)
    # Initialize a batch
    batch = model_train.Batch(src_tender, trg_tender, pad=-1)
    # debug("batch", batch)
     
    for epoch in range(500):
        transformer.mainlogger.info("Epoch: %s", epoch + 1)
        # # Construct loss object (this can be changed to model_train.MultiGPULossCompute)
        # loss = transformer.SimpleLossCompute(model.generator, loss_criterion, model_opt)
        # Simpler version
        loss = loss_criterion
        # Modify in_progress repeatedly through run_epoch()
        in_progress = run_epoch(model, size_cont, pattern, input_image, saveName, V_src, in_progress, batch_size, loss, batch)
    
    

