#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:20:33 2021
@author: mbaxszlh
"""
import torch
from torch import nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os 
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image as imwrite
#import itertools
#from skimage.morphology import skeletonize
from torch.autograd import Variable
from copy import deepcopy
from scipy import signal
import PIL.Image as PIL_Image


from model import *
from model_chapter3 import *

def main(folder, train_data, test_data, epoch=150, parallel_num=3, angle_num=4, k_size=9, lambda_=0.1, warm_up=False, identical_init=True, l2_norm=True, return_model=True, resume=False):
    
    '''
    Test Images are Resized of folds of 12 !
    '''
    lr_resume = 0
    if resume and lr_resume<0:
        raise ValueError('Please give lr resume value.')
    if not resume and lr_resume!=0:
        print('Please restore lr resume value.')
        lr_resume = 0
        
    print('task: '+folder)

    batch_size = 32
    volume = len(train_data) + len(test_data)
    print('training volume: ', volume)
    train_data = WrappedDataLoader(get_data(train_data, batch_size), preprocess)
    test_data = WrappedDataLoader(get_data(test_data, batch_size), preprocess)
    
    if not resume:
        model = MASC_Model(angle_num, parallel_num, k_size, identical=identical_init)
        torch.save(model, folder + '/init_model')
    else:
        model = torch.load(folder+'/model')
        
    torch.save(model, folder + '/init_model')
    print('model parameters: ', get_n_params(model))
    if model.MASC_Block1.MAC.messageIntegration in ['cosine', 'diffCosine']:
        print('GConv Block2 b2: '+str(model.MASC_Block1.MAC.b2[0].data))
    # print(model.GConv4.conv_w[0][0])
    loss_func = nn.BCELoss()
    loss_func2 = nn.BCELoss()
    # loss_func2 = nn.MSELoss()
    if resume:
        opt = torch.optim.Adam(model.parameters(), lr = 0.001*(0.5**lr_resume))
    else:
        opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    img = cv2.imread('data/testset/Image_11L.jpg')
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    img = torch.from_numpy(img)
    img = img.transpose(0, 2).transpose(1, 2)
    img = img.view(1, *img.shape).float()
    img, ori_h, ori_w = code_testimg(img)
    
    
    #train
    loss_recorder, model = fit(model, epoch, loss_func, loss_func2, lambda_, opt, train_data ,test_data, folder, img, warm_up=warm_up, l2_norm=l2_norm, resume=resume)
    if model.MASC_Block1.MAC.messageIntegration in ['cosine', 'diffCosine']:
        print('MASC_Block1 b2: '+str(model.GC_Block1.GConv.b2[0].data))
    # print(model.GConv4.conv_w[0][0])
    torch.save(model, folder + '/model')
    np.save(folder + '/log', loss_recorder)
    
    a = model(img, l2_norm)
    index = 0
    for i in a[1:4]:
        pmap = to_array(i)[0, 0]
        pmap = 255*(pmap/pmap.max())
        cv2.imwrite(folder + '/pmap'+str(index)+'.png', pmap)
        index += 1
    index = 0
    for i in a[4]:
        cv2.imwrite(folder+'/index'+str(index)+'_ep'+str(epoch)+'.png', to_array(i[0,0]))
        index += 1
    imwrite(a[0][0,0]//1, folder + '/out.png') 
    #
#    model = torch.load('model')
    ##loss_func = nn.BCELoss()
    ##opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    ##train
    ##loss_recorder, model = fit(model, 100, loss_func, opt, train_data ,test_data)
    test_list = os.listdir('data/testset/')
    os.system("mkdir "+folder+"/test")
    for i in test_list:
        print(i)
        img = cv2.imread('data/testset/'+i)[:, :992]
        img = cv2.resize(img,None,fx=0.5,fy=0.5)
        img = torch.from_numpy(img)
        img = img.transpose(0, 2).transpose(1, 2)
        img = img.view(1, *img.shape).float()
        img, ori_h, ori_w = code_testimg(img)
        img = to_array(uncode_testimg(model(img, l2_norm)[0], ori_h, ori_w)[0, 0]*255)
        cv2.imwrite(folder+'/test/'+i, img)
    if return_model == True:
        return model
    else:
        return 
    
num_fold = 5
parallel_num = 2
k_size = 9
# theta in [0, 2pi)
angle_num = 16
for fold in range(num_fold):
    os.system('mkdir models/MASC_p{}a{}k{}l2_fold{}'.format(parallel_num, angle_num, k_size, fold))
    img = np.load('data/training_img.npy')
    label = np.load('data/training_label.npy')
    img2 = np.load('data/testing_img.npy')
    label2 = np.load('data/testing_label.npy')
 
    x_data, y_data = map(torch.tensor, (img, label))
    x_data = x_data.type('torch.FloatTensor')
    y_data = y_data.type('torch.FloatTensor')
    trainset = TensorDataset(x_data, y_data)
    
    x_data, y_data = map(torch.tensor, (img2, label2))
    x_data = x_data.type('torch.FloatTensor')
    y_data = y_data.type('torch.FloatTensor')
    testset = TensorDataset(x_data, y_data)
    
    Reg_mode = 3
    # temp = main('5MASC/temp', trainset, testset, epoch=3, parallel_num=2, angle_num=8, k_size=5, warm_up=True, identical_init=True, l2_norm=True, return_model=True)
    model = main('models/MASC_p{}a{}k{}l2_fold{}'.format(parallel_num, angle_num, k_size, fold), trainset, testset, epoch=150, parallel_num=parallel_num, angle_num=angle_num, k_size=k_size, lambda_=0.01, warm_up=False, identical_init=True, l2_norm=True, return_model=True)
    


GConv_visual(model)
convW_visual(model)
correlation_visual(model)