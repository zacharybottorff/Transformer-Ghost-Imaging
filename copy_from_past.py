from ast import Num
import transformer_badge as transformer
import model_train_construct as model_train
from typing import Pattern
from torch.utils.data import Dataset,dataloader,TensorDataset
from torch.autograd import Variable
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST
import math



# class Batch(object):
#     """
#     Object for holding a batch of data with mask during training.
#     """

#     def __init__(self, src, trg=None, pad=0):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         self.src_mask = self.src_mask.cuda()
#         if trg is not None:
#             trg = trg.to(int)
#             self.trg = trg[:, :-1]
#             self.trg_y = trg[:, 1:]
#             self.trg_mask = self.make_std_mask(self.trg, pad)
#             # self.ntokens = (self.trg_y != pad).sum().item()
#             self.ntokens = (self.trg_y.shape[1])
#             # self.ntokens = self.trg_y.shape[1]

#     @staticmethod
#     def make_std_mask(tgt, pad):
#         """
#         Create a mask to hide padding and future words.
#         """

#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         aba = subsequent_mask(tgt.size(-1)).cuda()
#         tgt_mask = tgt_mask & aba
#         return tgt_mask

class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, fileFolder="D:\\study\\PatternDL\\python\\data"
                 , imsize=[112, 112]):
        fileName = os.path.join(
            fileFolder, 'training_input.npy')
        if not os.path.isfile(fileName):
            raise IOError(' This file doesn\'t exist {:} '.format(fileName))
        # loading data
        self.x_data = np.load(fileName)
        self.len = np.size(self.x_data, 0)
        self.transform = trasnFcn(imsize)
        if len(imsize) == 1:
            imsize = [imsize, imsize]

    def __getitem__(self, index):
        img = Image.fromarray(self.x_data[index, :, :], mode='L')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.len

def LoadModel(model,SaveModelFile):
    state  = torch.load(os.path.join(SaveModelFile,'Modelpara.pth'))
    model.load_state_dict(state['net'])
    epoch               = state['epoch']
    epochTrainingLoss   = state['TrainingLosses']
    MINloss             = state['MINloss']
    return model,epoch,epochTrainingLoss,MINloss


def trasnFcn(imsize=[54, 98], datamean=0.5,
             datastd=0.5, Moving=True):
    resizeIMG = transforms.Resize(size=imsize)
    rotate = transforms.RandomRotation(180)
    totensor = transforms.ToTensor()
    transform = transforms.Compose([
        resizeIMG,
        rotate,
        transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
        totensor,
        # normalise,
    ])
    if Moving:
        transform = transforms.Compose([
            resizeIMG,
            rotate,
            transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
            totensor,
            transforms.RandomCrop(imsize, padding=(56, 56))
            # normalise,
        ])
    return transform

def LoadData(MNISTsaveFolder, imsize=[28, 28], train=True, batch_size=32, num_works=0, DataSetName=MNIST):
    # original image size is [28,28]
    # data_set = DealDataset(imsize=imsize)
    datamean = 0.5
    datastd = 0.5
    Trans = trasnFcn(imsize, datamean=datamean, datastd=datastd)
    if train:
        if isinstance(DataSetName, tuple):
            data_set = []
            for SUBsetName in DataSetName:
                data_set.append(SUBsetName(root=MNISTsaveFolder, train=True, transform=Trans, download=True))
            data_set = ConcatDataset(data_set)
        else:
            data_set = DataSetName(root=MNISTsaveFolder, train=True, transform=Trans, download=True)
    else:
        if isinstance(DataSetName, tuple):
            data_set = []
            for SUBsetName in DataSetName:
                data_set.append(SUBsetName(root=MNISTsaveFolder, train=False, transform=Trans, download=True))
            data_set = ConcatDataset(data_set)
        else:
            data_set = DataSetName(root=MNISTsaveFolder, train=False, transform=Trans)

    dataLoader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=num_works,
                            drop_last=True)
    return dataLoader

def npySave(FileName,tensor,SaveModelFile):
    np.save(os.path.join(SaveModelFile,FileName),tensor.to('cpu').detach().numpy())


# def trg_dealwith(input_image, imsize):
#     arrange_likeu = torch.arange(1, imsize[0] * imsize[0] + 1)
#     input_image = input_image.reshape(input_image.shape[0], imsize[0] * imsize[0])
#     trg = input_image * arrange_likeu
#     trg = trg.squeeze()
#     # print("Line 420,trg.shape ",trg.shape) [32,900]
#     find_max_dim = torch.count_nonzero(trg,dim=1).max()
#     trg_batch = torch.zeros(trg.shape[0],find_max_dim)
#     index_x = 0
#     while (index_x != trg.shape[0]):
#         trg_pice = trg[index_x, :]
#         trg_nonzero = trg_pice.nonzero()
#         trg_pice = trg_pice[trg_nonzero].squeeze()
#         trg_batch[index_x,0:trg_pice.shape[0]] = trg_pice
#         index_x += 1
   
#     trg_pice_zero = torch.zeros(trg_batch.shape[0],trg_batch.shape[1]+1)
#     trg_pice_zero[:,1:] = trg_batch
#     trg_pice_zero = trg_pice_zero.cuda()
#     # print("Line 434,trg_pice_zero.shape ",trg_pice_zero.shape)
#     return trg_pice_zero

def show_result(trg,ColRow_num,cell_length,imsize,batches):

    trg_tendere = torch.clone(trg)
    for_show = torch.zeros(batches,imsize[0], imsize[0])
    for pic_num in range(batches):#here to get a whole pic
        index_smallOne = 0
        for col in range(ColRow_num):
            for row in range(ColRow_num):# here to get each small pic
                index_conut = (cell_length*cell_length-1)
                for col_least in range(cell_length):
                    for row_least in range(cell_length):#here to get the count of each small cell
                        # trg[pic_num, index_smallOne] += (pow(2,index_conut)*input_image[pic_num,cell_length*col+col_least,cell_length*row+row_least])
                        for_show[pic_num, cell_length * col + (cell_length-1-col_least), cell_length * row + (cell_length-1-row_least)] = (trg_tendere[pic_num, index_smallOne]>=pow(2,index_conut))
                        trg_tendere[pic_num, index_smallOne] -= int((pow(2,index_conut)*for_show[pic_num, cell_length * col + (cell_length-1-col_least), cell_length * row + (cell_length-1-row_least)]))
                        index_conut -= 1
                index_smallOne += 1
    plt.imshow(for_show[0,:,:])
    # print(for_show[1,:,:])
    plt.show()

# def greedy_decode(model, src, src_mask, trg, start_symbol):
#     src = src[0:1, :]
#     max_length = trg[0:1,:].nonzero().shape[0]
#     print("max_length.lenghth",max_length)
#     src_mask = src_mask[0:1, 0:1, :]
#     print("src",src.shape)
#     print("src_mask",src_mask.shape)
#     memory = model.module.encode(src, src_mask)
#     # memory = [2,62,512]
#     ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).cuda()
#     # ys = [1,1]
#     for i in range(max_length - 1):
#         out = model.module.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
#         prob = model.module.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()
#         ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long).fill_(next_word).cuda()], dim=1)
#     return ys






# def greedy_show(model, src, src_mask, trg,SaveModelFile,index,size_cont):
#   for_show = greedy_decode(model, src, src_mask, trg, start_symbol=1)
#   result = for_show.reshape([1,for_show.shape[0]*for_show.shape[1] ])
#   result = result.cpu()
#   trg = trg.cpu()
#   g = torch.zeros(size_cont*size_cont)
#   g[result.squeeze () - 1] = 1
#   g = g.reshape(size_cont,size_cont)

#   b = torch.zeros(size_cont*size_cont)
#   b[(trg[0,:] - 1)] = 1
#   b = b.reshape(size_cont,size_cont)

#   if (index == 0):
#       b = np.expand_dims(b, axis=0)
#       g = np.expand_dims(g, axis=0)
#       last_one = np.append(b, g, axis=0)
#       np.save(os.path.join(SaveModelFile, 'greedy_result.npy'), last_one)
#   else:
#       last_one = np.load(os.path.join(SaveModelFile, 'greedy_result.npy'))
#       b = np.expand_dims(b, axis=0)
#       g = np.expand_dims(g, axis=0)
#       last_one = np.append(last_one, b, axis=0)
#       last_one = np.append(last_one, g, axis=0)
#       np.save(os.path.join(SaveModelFile, 'greedy_result.npy'), last_one)
#   index = index + 1
#   return index

# def run_epoch(model, loss_compute,size_cont,SaveModelFile,V2):
#     """
#     Standard Training and Logging Function
#     """
#     imsize = [32]
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     index = 0
#     input_image = np.load(os.path.join(SaveModelFile, 'image_number_32.npy'))

#     PatternOrigin = np.load(os.path.join(SaveModelFile, 'simulation_rayleigh_src_75.npy'))
#     PatternOrigin = torch.from_numpy(PatternOrigin)
#     PatternOrigin = PatternOrigin.cuda()
#     src_tender = PatternOrigin
#     trg_tender = trg_dealwith(input_image,imsize)
#     npySave("src.npy",src_tender,SaveModelFile)
#     npySave("trg.npy",trg_tender,SaveModelFile)
#     batch = Batch(src_tender, trg_tender, 0)
#     index = greedy_show(model, batch.src, batch.src_mask, batch.trg ,SaveModelFile,index,size_cont)
#     out = model.forward(batch.src[1:,:], batch.trg[1:,:], batch.src_mask[1:,:], batch.trg_mask[1:,:])
#     loss = loss_compute(out, batch.trg_y[1:,:], batch.ntokens)
#     total_loss += loss
#     total_tokens += batch.ntokens
#     tokens += batch.ntokens
#     return total_loss / total_tokens


# def src_dealwith(img_ori, pattern,V2):
#     """
#         generate ghost images from image_ori
#         img_ori : Tensor [batch_size, in_channel, [imsize]]
#         pattern : Tensor [batch_size, Number_Pattern, [imsize]]
#         Number_Pattern : int
#         imsize : int or turple with length of 2
#         CGIpic is normalized, and target is range from 0 to 255
#         the CGIpic is normalized to mean value 0.25 0.2891

#         Other variables in this function
#         I : intensity [bacth_size, Number_Pattern]
#     """
#     I = pattern
#     I_min = pattern.min()
#     I = I - I_min
#     I_max,I_index = torch.max(I,1)
#     I_max =I_max.reshape(I.shape[0],1)
#     I = I/(I_max+1)*900
#     I = I.int()
#     return I

def check_stat(pattern_3d, size_len=112, show_PIC=True, check_fft=True, check_pdf=True, check_g2=False, check_var=True):
    if (len(np.shape(pattern_3d)) == 3):
        if show_PIC:
            ave = np.zeros((size_len, size_len))
            for i in range(size_len):
                for j in range(size_len):
                    ave[i, j] = np.average(pattern_3d[0:, i, j])
            plt.imshow(ave)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("Pic_Avarage")
            plt.colorbar()
            plt.show()
            plt.imshow(pattern_3d[0, :, :])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("show_One_Pic")
            plt.colorbar()
            plt.show()

        if check_fft:
            f = np.fft.fft2(pattern_3d[0])
            fshift = np.fft.fftshift(f)
            res = np.log(np.abs(fshift))
            plt.imshow(res)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()
            plt.title("FFT")
            plt.show()

        if check_pdf:
            ave_1dim = pattern_3d[0].reshape((size_len * size_len))
            maxx = ave_1dim.max()
            ave_1dim = (ave_1dim / maxx) * 255
            plt.hist(ave_1dim, 50)
            plt.title("PDF")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if check_g2:
            loadData_g2 = pattern_3d
            PI = np.zeros((size_len, size_len))
            P = np.zeros((size_len, size_len))
            N = loadData_g2.shape[0]
            middle = int(size_len / 2)
            for i in range(N):
                PI = PI + loadData_g2[i, :, :] * loadData_g2[i, middle, middle]
                P = P + loadData_g2[i, :, :]
            PI = PI / N
            P = P / N
            I = P[middle, middle]
            g2 = PI / P / I - 1
            plt.imshow(g2)
            plt.title("g2")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()
            plt.show()

        if check_var:
            reshape_pattern0 = pattern_3d[0].reshape((size_len * size_len))
            x = np.arange(1, reshape_pattern0.shape[0] + 1)
            plt.plot(x, reshape_pattern0, color='red')
            var = np.var(np.array(reshape_pattern0))
            plt.title('***VAR*** = ' + str(var), fontsize=20, color="red")
            plt.show()
    else:
        if (len(np.shape(pattern_3d)) == 2):
            if check_pdf:
                ave_1dim = pattern_3d[0]
                maxx = ave_1dim.max()
                ave_1dim = (ave_1dim / maxx) * 255
                plt.hist(ave_1dim, 50)
                plt.title("PDF")
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

            if check_var:
                reshape_pattern0 = pattern_3d[0]
                x = np.arange(1, reshape_pattern0.shape[0] + 1)
                plt.plot(x, reshape_pattern0, color='red')
                var = np.var(np.array(reshape_pattern0))
                plt.title('***VAR*** = ' + str(var), fontsize=20, color="red")
                plt.show()

