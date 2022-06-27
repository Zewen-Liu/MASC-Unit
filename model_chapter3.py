#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5.6 MASC



    v3
    1. Based on the paper version, give more variance to the Gabor initialisation.
    2. And also using puramid pooling to enlarge the pracitial receptive field. 
    (to replace the kernel scaling scheme and low-resolution layer)
    3. Using difference max margin (message integration method, which used to be 
    maximum selection/squared sum/umnormalised softemax)
    4. GConv node is shared among each GConv blocks, enhance the training effect,
    though an interesting phenomenon was obeserved. The Gabor kernel tends to evolude
    to all 0s or fullfill the window, when the conv_w learned the direction-sensitive
    pattern (mirror according to an axis across the origin).
    5. Fixed lambda in Gabor generator
    6. Choose if initialise conv_w as identical as Gab
    7. Can training with warm-up scheme, where the input map to any GConv node 
    will be trained in a warm-up way for 10 epochs each. Adujust the term 
    coefficient in diff_max_marginal to 2.
    8.Enhance the weight given to the maximum channel.
    Use max to replace the soft-max function for processing the pyramid result.    
    9.Use the weight method of 3.7, use product combination in packing kernel.
    10. Increase weight method of 3.7 from 2 to 3
    Cancel l2 feature=normlisation in GC_node.
    combine gc-node output  mutiply with the residuals and feature channels, used
    to be concatenation
    ---------------------------------------------------
    1. The convolution space of GC-node and input data is a flower-like manifold.
    Each petal is the manifold of each direction.The tip of a petal represents
    the best-matching case.
    Calculate the optimal direction pattern, and compare the GC-node output
    with the optimal pattern by vector cosine distance. The closer the better.
    
    Use symmetric max to substitute the simple max, as the input feature map sign
    could be flipped. The function is defined as,
        max(max, |min|)*original sign.
    
    2.
    Cancel kernel centrialisation.
    Divide l2 in GC-convolution, multiply l2 back using a smaller window.
    
    --vehicle: made for testing the analysing capacity of steerable conv.
        1. Use stronger denoise sampler, k=1.
        2. Replace l3 and p2 by 8-2-8 GConv-Block.
        3. Use smaller k_size, tested 3 and 5.
        4. Not use GConv-maps as attention layers but as normal feature maps.
        5. Use mono max not symmetric max to speed up training.
        6. Only use Gab in initilising conv_w, and not as original GC-packing.
        7. More room for conv_w Gabor-itialisation (updatable lambda in Gabor-Generator).
        
        -2.
        1. add new pattern regularisation to loss func, with exp
        
        -3.
        Identical initialisation with amplituded noise, amplitude=Gab.amplitude.
        
        -4
        Use normalised cosine function (0, 1) in external regularisation optimisation term.
        Apply 1/8 amplituded noise in conv_w initialisation.
        
        -5
        Use relu(cosine) regularisation optimisation term, as it is hard to tune
        all the directions, focus on the close ones.
        
        -6
        (-)Use KL regularise optimal pattern and use strech loss to generate intensity
        when comparing the optimal pattern and convolutional result.
        
        (?)New l2-norm, remove norm2 in GConv-layer and then multiply with the persudo 
        intensity map(residual).
        
        -7
        Use stretch searching message function.
        Adjust the packing weights on the Gab for non-id initilisation.
        Roll back to vanilla l2-difference denosing, use k_size = 3 as compensation
        concatenate the conv stream and residual stream, 
        
        Rewrite get_optimal_pattern() including replace .reshape() to .view().
        
        Turn noise level of id_init mode to 1/3 of Gab amplitude.
        
        norm(x) and norm(m) used in cosine-searching function of GC_node.
        Use c*((ones-eye)*M_ori*M).sum() as Reg_loss
        
        x_sigma     >= k/2
        y_sigma*3   <= k/6
        gamma = x_sigma/y_sigma
        gamma >= 9
        
        Use output = cos(exp(Mi), exp(V))
    
    v5
    -1. The version used in the paper "MASC".
    
    -2. Exp in response shaping and using inductive bias term. 
        only put exp(bias) on shaped response (exp degree=2): 
            [0.72813226 0.97785885 0.78269428 0.79825872 0.95628581 0.97462591]
            bias~[-7.2, -1.6], mostly [-3.6, -2]
        only put exp(bias) after v and m (exp degree=2):
            [0.72350017 0.97728676 0.78466268 0.79762357 0.95599454 0.9742655 ]
            bias~[-2.1, -0.13], mostly ~[-1, -0.3]
        put exp(bias_1) and exp(bias_2) on shaped reshape and v&m, and learnable exp degree:
            [0.72424027 0.97687504 0.79015574 0.79966288 0.95623552 0.97404473]
            exp_degree mean=4.90, std=3.54
    
    -3. 
        Power information integration with MASC intensity map and conv features
        Use mean(m) to scale v not norm(v), this is both theoretical better and
        proved by experiment. It allows the intensity disparity between the target 
        and non-target areas.
        
    -4. Second Order Pool combine with features with exp_messae manner
        -Another choise for message integration, the reorder mode. Reorder MAC 
        feature vector v with the strongest first rest order remain. Then combine 
        the reordered features with a 1x1 conv layer. It doesnot work as well as 
        normal MASC.
        
    -5. 0.Improved Pool2nd: Second Order Pool with selected mean base.
        1.Modify Response shaping as exp(c*v) * exp(c*m), was exp(log(c)*v)*exp(log(c)*m).
          Add new diffCosine message integration function(by default still cosine function).
          
          add scale_level = 1, now set (1, 2, 3, 4).
          
          file: 551_p2a8k5l2_diffCosine_all2ndexpadd_geo_8000_temp1
    
    -6 Featurise index map.
        1.Use momentum when updating M(optimal_pattern) according to Moco_v1, the 
          intial ratio is set to be 0.99.
          
          Featurised index map.(found very useful in pirot experiments)
                        mean       std
            f1        0.800124  0.007098
            acc2      0.955664  0.000930
            roc_auc2  0.975084  0.000848
        2.Smooth(Gassian) the featurised index map.
        3.Combine directional feature.
        4.Use grouped conv to combine the directional featurise index maps from
          parallel units. Refine these feature maps particularly. Then summerise
          all the information to parallel number channels.
                        mean           std
            dice2     0.723067  0.000000e+00
            sp        0.978969  0.000000e+00
            se        0.771116  0.000000e+00
            f1        0.794823  1.110223e-16
            acc2      0.955993  1.110223e-16
            roc_auc2  0.972577  0.000000e+00
          Seems not working, roll back to 563.
        5.Based on .3, use mean_pool to scale featurised index map.
    
    -7 1.Pattern loss mode 4 (rotational secure term)
        minimise the std among the G&W correlation, which is equal to minimise
        pattern_loss4 = e ^ (std(angle_corr) - c)
        angle_corr    = sum(G*W, dim=(1, 2, 3)) = sum(K, dim=(1, 2, 3))
        given K       = G*W,
        where c is a constant number, c = 2 tentatively.
        The rotational effect used to be protected by the latent rotational patterns
        in the dataset, and the even numbered pattern rotations helps secured this
        effect. In this version, come up with a simple idea to provid hard secure on
        pattern rotation, this is achieved by addtional correlation management.
        
        Assume Gabor filters are generally ideally steerable. The correlation between 
        Conv and Gabor should be isotropic if the sythetic patterns K=G*W are
        also ideally steerable. So, inductively, the model should have the variance
        be smallg. These are the ideas behind the equations given at the beginning of 
        the updating comments.
       2. Use k_size**2 Gabor patterns to calibrate the conv part. 
       3. Extend rotation range from Pi (90rot) to 2Pi (90rot & 180rot).
       4. Use different input for each parallel MASC unit.
    
    -8 1.Average M
        According to the conception about a universal solution of rotatable kernel,
        update M as averaged response on the directions. It is critical for forming
        a precisely rotated effect as the M matrix in the ideal case should have 
        equal response on all (theta_i, theta_i + k), i is any directional index,
        and k is a constant phase offset. In M matrix, this means all the elements
        on the diagonal with k phase offset should equal to the same number if 
        the kernels are precisely rotated.
        
        Rotational secure term applies correlation measurement on G_i_k and W_i,
        variable k is the phase offset here. it can provide more variation on 
        correlation base and improve the rotational score accuracy. The k's defined
        range is the every input angle interval (2pi/angle_num) in half pi.
       
      4.Relative featurisation
        Index features are folded and adjusted with the central value. Such as,
        a window of index to be [2 ,3 , 4], then the folded and adjusted feature
        vector to be [[-1, 0, 1]]. 
        
        
    
      6. Add direction lock to mac.loss_pattern(), not only base kernels, but
        base kernels and their transpose are involved in the phase-offset correlation
        estimation.
        
        Fix kernel order error, rot90 does an anti-clockwise rotation, but was thought
        as clockwise. It was used in arranging directinal kernel order. Now let
        the parameter kernel.rot90(t=3).
        
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
#from torchvision.transforms import Normalize


import time

kernel_size = 5
    
def norm_kernel(kernel, centralise=False):
    b, c, k1, k2 = kernel.shape
    if centralise:
        kernel = kernel - kernel.mean(dim = (2,3)).view(b, c, 1, 1)
    return kernel/torch.norm(kernel, dim=(2,3)).view(b, c, 1, 1)


class exp_message(nn.Module):

    '''
    Porject x1 input to (0, 2) with 2*sigmoid(x1).
    For the elements in (0, 1), exponential function has a negtive derivative;
    For the elements in (1, 2), exponential function has a positive detivative.
    By leverage this bi-directional character, the message is transformed to
    a beatiful equation
            EMess(x1, x2) = f(x1)^x2,
            f(x) = 2*sigmoid(x)
    We call x1 as root feature, x2 as guidance.
    '''
    def __init__(self, b1=True, b2=True):
        super().__init__()
        if b1:
            self.b1 = nn.Parameter(torch.randn(1).uniform_(-0.75, 0.1))
        else:
            self.b1 = 0
        if b2:
            self.b2 = nn.Parameter(torch.randn(1).uniform_(-0.1, 0.1))
        else:
            self.b2 = 0
        self.c = nn.Parameter(torch.randn(1).uniform_(1.5, 2.5)) 
    
    def forward(self, x1, x2):
        # print(x1.max().detach(), (torch.sigmoid(x1.max())*2).detach(), x2.max().detach(),
        #        x1.min().detach(), (torch.sigmoid(x1.min())*2).detach(), x2.min().detach())
        return (torch.sigmoid(x1/self.c+self.b1)*2+0.00001).pow(x2+self.b2)

class selected_meanPool(nn.Module): 
    '''
    Calculate mean value among a selected set of neighbors determined by mask.
    '''
    def __init__(self, k_size, dilate=1, stride=1):
        super().__init__()
        self.stride = stride
        self.pooling_2nd_k = k_size
        self.unfold_dilation = dilate
        # Pad the input feature later inside the function
        self.unfold = nn.Unfold(self.pooling_2nd_k, stride=self.stride, dilation=self.unfold_dilation, padding=0)
        
    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        if mask == None:
            mask = torch.ones(((b, 1, h, w)))
            
        if (x.shape[-2:] != mask.shape[-2:]) or (mask.shape[1] != 1):
            raise ValueError('The shapes of feature and mask do not match.')
        
        x_unfold = self.unfold(F.pad(x, [int(self.pooling_2nd_k/2)*self.unfold_dilation]*4, mode='reflect')).view((b, c, self.pooling_2nd_k**2, h, w))
        
        mask_repeat = mask.view(b, 1, 1, h, w).repeat(1, 1, self.pooling_2nd_k**2, 1, 1)
        mask_unfold = (self.unfold(F.pad(mask, [int(self.pooling_2nd_k/2)*self.unfold_dilation]*4, mode='reflect')).view((b, 1, self.pooling_2nd_k**2, h, w)) == mask_repeat)*1
        mask_unfold = mask_unfold.repeat(1, c, 1, 1, 1)
        temp = (mask_unfold.sum(dim=(1, 2))/c).view(b, 1, h, w)
        temp[temp==0] = 1
        return ((mask_unfold*x_unfold).sum(dim=2)/temp).view((b, c, h, w))


class indexFeaturize(nn.Module):
    '''
    Project index maps to a angle_num-channels space
    '''
    def __init__(self, angle_num, parallel_num, gaussian_size=5):
        super().__init__()

        self.angle_num = angle_num
        self.parallel_num = parallel_num
        self.gaussian_size = gaussian_size
        self.gaussian_kernel = torch.tensor(np.outer(signal.gaussian(self.gaussian_size, 1), signal.gaussian(self.gaussian_size, 1)).astype('float32'))
        self.gaussian_kernel = self.gaussian_kernel.view((1, 1, self.gaussian_size, self.gaussian_size)).repeat(self.angle_num*self.parallel_num, 1, 1, 1)
        self.conv1 = nn.Conv2d(self.angle_num*self.parallel_num, self.parallel_num, kernel_size=3, stride=1, padding=1, groups=self.parallel_num, padding_mode='reflect')
        self.conv2 = nn.Conv2d(self.parallel_num, self.parallel_num, kernel_size=3, stride=1, padding=1, groups=self.parallel_num, padding_mode='reflect')
        self.c = nn.Parameter(torch.randn(1).uniform_(self.angle_num-0.1, self.angle_num+0.1))
        
    def forward(self, index_map, x, scaleLevel, smooth=False):
        # print(self.conv1.weight.shape)
        b, parallel_num, h, w = index_map.shape
        # if parallel_num != self.parallel_num:
        if parallel_num != self.parallel_num:
            raise ValueError('Parameter parallel number({}) does not match input index map\'s parallel number({})'.format(self.parallel_num, parallel_num))
        featurised_map = -torch.ones((b, self.angle_num*self.parallel_num, h, w))
        # featurised_map = torch.zeros((b, self.angle_num*self.parallel_num, h, w))
        
        # featurised_map = F.conv2d(featurised_map, self.gaussian_kernel, groups=self.angle_num*self.parallel_num, padding=self.gaussian_size//2)
        for j in range(self.angle_num):
            temp = (index_map==j)
            # featurised_map[:, j::self.angle_num] = x*temp*self.angle_num + featurised_map[:, j::self.angle_num]*(~temp)
            
            # Amplify the signal strength by the direction number. Otherwise, 
            # the intensities would be rather small after the convolutional 
            # processes.
            featurised_map[:, j::self.angle_num] = x*temp*self.c + featurised_map[:, j::self.angle_num]*(~temp)
        if smooth:
            featurised_map = F.conv2d(featurised_map, self.gaussian_kernel, groups=self.angle_num*self.parallel_num, padding=self.gaussian_size//2)
        
        feature_maps = []
        for i in range(len(scaleLevel)):
            temp = F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(F.avg_pool2d(featurised_map, kernel_size=scaleLevel[i], stride=scaleLevel[i])))))
            # temp = F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(F.avg_pool2d(featurised_map*(1+1/(scaleLevel[i]**2))-1/(scaleLevel[i]**2), kernel_size=scaleLevel[i], stride=scaleLevel[i])))))
            # print(temp.shape)
            feature_maps.append(F.interpolate(temp, size=(h, w), mode='bilinear'))
        return torch.stack(feature_maps, dim=1).max(dim=1)[0]
    
class indexFeaturize_relative(nn.Module):
    '''
    Project index maps to a angle_num-channels space
    '''
    def __init__(self, angle_num, parallel_num, gaussian_size=5):
        super().__init__()

        self.angle_num = angle_num
        self.parallel_num = parallel_num
        self.gaussian_size = gaussian_size
        self.gaussian_kernel = torch.tensor(np.outer(signal.gaussian(self.gaussian_size, 1), signal.gaussian(self.gaussian_size, 1)).astype('float32'))
        self.gaussian_kernel = self.gaussian_kernel.view((1, 1, self.gaussian_size, self.gaussian_size)).repeat(self.parallel_num, 1, 1, 1)
        self.conv1 = nn.Conv2d(self.parallel_num, self.parallel_num, kernel_size=5, stride=1, padding=2, groups=self.parallel_num, padding_mode='reflect')
        self.conv2 = nn.Conv2d(self.parallel_num, self.parallel_num*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(self.parallel_num*2, self.parallel_num, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.c = nn.Parameter(torch.randn(1).uniform_(self.angle_num-0.1, self.angle_num+0.1))
        
    def forward(self, index_map, x, scaleLevel, smooth=False):
        # print(self.conv1.weight.shape)
        b, parallel_num, h, w = index_map.shape
        # if parallel_num != self.parallel_num:
        if parallel_num != self.parallel_num:
            raise ValueError('Parameter parallel number({}) does not match input index map\'s parallel number({})'.format(self.parallel_num, parallel_num))   
        featurised_map = self.conv1(index_map)
        # print(featurised_map.shape, self.conv1.weight.shape)
        for i in range(parallel_num):
            featurised_map[:, i] = featurised_map[:, i] - index_map[:, i]*self.conv1.weight[i, 0].sum()
        
        featurised_map = self.c*x*featurised_map
        
        if smooth:
            featurised_map = F.conv2d(featurised_map, self.gaussian_kernel, groups=self.angle_num*self.parallel_num, padding=self.gaussian_size//2)
        
        feature_maps = []
        for i in range(len(scaleLevel)):
            temp = F.leaky_relu(self.conv3(F.leaky_relu(self.conv2(F.avg_pool2d(featurised_map, kernel_size=scaleLevel[i], stride=scaleLevel[i])))))
            # temp = F.leaky_relu(self.conv2(F.leaky_relu(self.conv1(F.avg_pool2d(featurised_map*(1+1/(scaleLevel[i]**2))-1/(scaleLevel[i]**2), kernel_size=scaleLevel[i], stride=scaleLevel[i])))))
            # print(temp.shape)
            feature_maps.append(F.interpolate(temp, size=(h, w), mode='bilinear'))
        return torch.stack(feature_maps, dim=1).max(dim=1)[0]
    
class Pooling2nd(nn.Module):
    '''
    Apply second_order pooling on input tensor x. The pooling_2nd can be regarded
    as conducting correlation on neighbors, neighbors are the selected positions 
    nearby constrained by a mask matrix.)
    '''
    def __init__(self, k_size, dilation, stride, base_mode='selective', corr_mode='plain'):
        super().__init__()
        basemodes = ['selective', 'plain', 'single']
        if base_mode not in basemodes:
            raise ValueError('Found input base_mode is {}, should be anyone in {}'.format(base_mode, basemodes))
        else:
            self.base_mode = base_mode
            
        corrmodes = ['selective', 'plain']
        if corr_mode not in corrmodes:
            raise ValueError('Found input corr_mode is {}, should be anyone in {}'.format(corr_mode, corrmodes))
        else:
            self.corr_mode = corr_mode
        self.stride = stride
        self.pooling_2nd_k = k_size
        self.unfold_dilation = dilation
        # Pad the input feature later inside the function
        self.unfold = nn.Unfold(self.pooling_2nd_k, stride=self.stride, dilation=self.unfold_dilation, padding=0)
        if self.base_mode == 'selective':
            self.pre_process = selected_meanPool(k_size=3)
        elif self.base_mode == 'plain':
            self.pre_process = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            
        self.b = nn.Parameter(torch.randn(1))
        
        
    def forward(self, x, mask=None, norm=True, geo=False):
        b, c, h, w = x.shape
        if self.base_mode == 'selective' and mask == None:
            raise ValueError('Pooling2nd: In the base selective mode, the mask variable needs to be given.')
            
        if self.corr_mode == 'plain' and mask == None:
            mask = torch.ones((b, 1, h, w))
        elif self.corr_mode != 'plain' and mask == None:
            raise ValueError('Pooling2nd: Have to give concrete mask in the corr_mode {}'.format(self.corr_mode))
            
        if (x.shape[-2:] != mask.shape[-2:]) or (mask.shape[1] != 1):
            raise ValueError('Pooling2nd: The shapes of feature and mask do not match.')
        
        if norm == True:
            x = x - torch.mean(x, dim=1, keepdim=True)
            x = x / (torch.norm(x, dim=1, keepdim=True)+0.00001)
        x_unfold = self.unfold(F.pad(x, [int(self.pooling_2nd_k/2)*self.unfold_dilation]*4, mode='reflect')).view((b, c, self.pooling_2nd_k**2, h, w))
        if self.base_mode == 'selective':
            x_repeat = self.pre_process(x, mask).view(b, c, 1, h, w).repeat(1, 1, self.pooling_2nd_k**2, 1, 1)
        elif self.base_mode == 'plain':
            x_repeat = self.pre_process(x).view(b, c, 1, h, w).repeat(1, 1, self.pooling_2nd_k**2, 1, 1)
        elif self.base_mode == 'single':
            x_repeat = x.view(b, c, 1, h, w).repeat(1, 1, self.pooling_2nd_k**2, 1, 1)
        else:
            raise ValueError('Pooling2nd: Base_mode {} seems not to be finished yet.'.format(self.base_mode))
        
        if self.corr_mode == 'selective':
            mask_repeat = mask.view(b, 1, 1, h, w).repeat(1, 1, self.pooling_2nd_k**2, 1, 1)
            mask_unfold = (self.unfold(F.pad(mask, [int(self.pooling_2nd_k/2)*self.unfold_dilation]*4, mode='reflect')).view((b, 1, self.pooling_2nd_k**2, h, w)) == mask_repeat)*1
            if self.base_mode == 'single':
                mask_unfold[:, :, int((self.pooling_2nd_k**2-1)/2), :, :] = 0
            mask_unfold = mask_unfold.repeat(1, c, 1, 1, 1)
        elif self.corr_mode == 'plain':
            mask_unfold = torch.ones((b, c, self.pooling_2nd_k**2, h, w))
        else:
            raise ValueError('Pooling2nd: Corr_mode {} seems not to be finished yet.'.format(self.base_mode))
        
        temp = mask_unfold.sum(dim=(1, 2))/c
        if self.corr_mode == 'single':
            temp += 1
        temp[temp==0] = 1
        if geo:
            out = (mask_unfold*x_unfold*x_repeat).sum(dim=1)+1
            out[out==0] = 1
            # print(out.prod(dim=1).max(), out.prod(dim=1).min(),out.shape)
            out = out.prod(dim=1).pow(1.0/temp)-1
        else:
            out = torch.exp((mask_unfold*x_unfold*x_repeat).sum(dim=(1))).sum(dim=(1))/temp
        # print(out.max(), out.min(),out.shape)
        return out.view((b, 1, h, w))

class GabLayer_DoG(nn.Module):
    def __init__(self, angle_num, k_size, parallel_num=2, in_groups=1, mode='cosine', identical=False, noise=True):
        super().__init__()
        all_modes = ['cosine', 'diffCosine', 'reorder', 'stretch', 'softMax', 'diffMax', 'monoMax']
        if mode not in all_modes:
            raise ValueError('Message integration function should be anyone of {}'.format(all_modes))
        self.messageIntegration = mode
        self.k_size = k_size
        self.scale_num = 1
        self.indentical_intial_convW = identical
        
        self.parallel_num = parallel_num
        self.angle_num = angle_num
        
        # self.w = nn.ParameterList([])
        # self.lamda = nn.ParameterList([])
        # self.w_l2 = torch.randn(1).uniform_(0.2, 0.5)
        self.Gab = nn.ModuleList([])
        self.conv_w = nn.ParameterList([])
        if mode in ['cosine', 'diffCosine']:
            self.mac_b = nn.ParameterList([])
            self.b2 = nn.ParameterList([])
        self.exp_degree = nn.ParameterList([])
        for i in range(parallel_num):   
            # self.lamda.append(nn.Parameter(torch.randn(1).uniform_(0, 0.5)))
            self.Gab.append(Generator(self.angle_num, self.k_size))
            if mode in ['cosine', 'diffCosine']:
                self.mac_b.append(nn.Parameter(torch.randn(1).uniform_(-0.001, -0.00001)))
                self.b2.append(nn.Parameter(torch.randn(1).uniform_(-1, -0.5)))
            self.exp_degree.append(nn.Parameter(torch.exp(torch.randn(1).uniform_(1.5, 2.5))))
            if self.indentical_intial_convW == True:
                base_num = int((self.angle_num/2+2)/4)
                forthPi = int(self.angle_num/4)
                base_stride = int(forthPi/base_num)
                if noise == True:
                    self.conv_w.append(nn.Parameter(self.norm_kernel(self.Gab[-1](scale = self.k_size)[1:forthPi:base_stride]+(self.Gab[-1].amplitude/12)*torch.rand(base_num, 1, self.k_size, self.k_size), centralise=True)))
                else:
                    self.conv_w.append(nn.Parameter(self.norm_kernel(self.Gab[-1](scale = self.k_size)[1:forthPi:base_stride], centralise=True)))
            else:
                if noise == True:
                    self.conv_w.append(nn.Parameter(self.norm_kernel(torch.ones(int((angle_num/2+2)/4), 1, self.k_size, self.k_size).float().uniform_(7/8, 1.0), centralise=True)))
                else:
                    self.conv_w.append(nn.Parameter(self.norm_kernel(torch.ones(int((angle_num/2+2)/4), 1, self.k_size, self.k_size).float(), centralise=True)))
            # temp = self.conv_w[-1]
            # temp[:, :, 0, 0] = temp[:, :, 0, 0]*1.25
            # self.conv_w[-1][:, :, 0, 0] = nn.Parameter(self.norm_kernel(temp))
            
        self.in_groups = in_groups

        self.bn = nn.BatchNorm2d(self.angle_num)
        self.padding = self.k_size//2
        # Used for correcting the filter order in direction extention
        self.idx = torch.LongTensor([i for i in range(self.conv_w[0].size(0)-1, -1, -1)])
        # Used for direction channel pattern searching
        self.optimal_pattern = self.get_optimal_pattern()
        
        
        # self.original_pattern = self.get_original_pattern()
        # self.original_pattern = (self.original_pattern+1)/2.01
        # self.original_pattern = self.get_p(self.original_pattern, dim=2, exp=False)
        
        if mode == 'reorder':
            self.reorder_combine = []
            for i in range(parallel_num):
                self.reorder_combine.append(nn.Conv2d(self.angle_num, 1, bias=True, kernel_size=1, stride=1, padding=0))

    def norm_kernel(self, kernel, centralise=False):
        '''
        Normalise the input kernel by dividing its norm-2.
        '''
        b, c, k1, k2 = kernel.shape
        if centralise:
            kernel = kernel - kernel.mean(dim = (2,3)).view(b, c, 1, 1)
        return kernel/(torch.norm(kernel, dim=(2,3)).view(b, c, 1, 1)+0.000001)
        
    def pack_kernel(self, pack_Gab=True):
        '''
        If identical, can pack Gabor integrated kernels for updating optimal pattern.
        Pack normalised kernel with its transpose, 90rotation, transpose.90rotation.
        
        Do not centrialised kernel, otherwise break the balance, as the kernel 
            of different direction have different positive-negative distribution.
        '''
        base_num = int((self.angle_num/2+2)/4)
        forthPi = int(self.angle_num/4)
        base_stride = int(forthPi/base_num)
        # 
        all_kernels = []
        for i in range(self.parallel_num):
            new_conv_w = self.conv_w[i]
            # if all(new_conv_w.std((2, 3))!=0):
            #     new_conv_w = self.norm_kernel(new_conv_w)
            '''
            Do not centrialised kernel, otherwise break the balance, as the kernel 
            of different direction have different positive-negative distribution.
            '''
            # new_kernel = self.norm_kernel(new_conv_w+self.lamda[i]*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride])
            new_kernel = self.norm_kernel(new_conv_w*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride], False)
            # new_kernel = self.norm_kernel(new_conv_w)
            # if pack_Gab and not self.indentical_intial_convW:
            #     new_kernel = self.norm_kernel(new_conv_w+2*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride]/self.Gab[i].amplitude)
            # if pack_Gab and not self.indentical_intial_convW:
            #     new_kernel = self.norm_kernel(new_conv_w*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride])
            # else:
                # new_kernel = self.norm_kernel(new_conv_w)
                # new_kernel = self.norm_kernel(new_conv_w*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride])
            new_kernel = torch.cat((new_kernel, new_kernel.transpose(2,3).index_select(0, self.idx)), dim=0) 
            new_kernel = torch.cat((new_kernel, torch.rot90(new_kernel, 3, dims=(2, 3))), dim=0)
            new_kernel = torch.cat((new_kernel, torch.rot90(new_kernel, 2, dims=(2, 3))), dim=0)
            all_kernels.append(new_kernel)
        return all_kernels
    
    def l2_layer(self, xb, kernel_size=3):
        '''
        Only on spatial not on depth.
        '''
        if kernel_size != 1:
            xb_padded = F.pad(xb, pad=(kernel_size//2, kernel_size//2)*2, mode='reflect')
            return F.lp_pool2d(xb_padded, norm_type=2, kernel_size=kernel_size, stride=1)
        else:
            return F.lp_pool2d(xb, norm_type=1, kernel_size=1, stride=1)
    
    def l2_layer3D(self, xb):
        '''
        Return channel-wise sqrt(sqaured sum)
        '''
        return torch.sqrt(xb.pow(2).sum(1, keepdim=True))
    
    
    def get_optimal_pattern(self, detach=True, averageM=True):
        '''
        Return the optimal patterm.
        Shape: parallel_num x angle_num x pattern_vector(angle_num, 1, 1)
        
        averagedM:
        Update M as averaged response on the directions. It is critical for forming
        a precisely rotated effect as the M matrix in the ideal case should have 
        equal response on all (theta_i, theta_i + k), i is any directional index,
        and k is a constant phase offset. In M matrix, this means all the elements
        on the diagonal with k phase offset should equal to the same number if 
        the kernels are precisely rotated.
        '''
        kernels = self.pack_kernel(self.indentical_intial_convW==False)
        optimal_pattern_tmp = []
        for k in range(self.parallel_num):
            temp = kernels[k].view((self.angle_num, self.k_size**2))
            optimal_pattern_tmp.append(torch.matmul(temp, temp.T).view(self.angle_num, self.angle_num, 1, 1))
        optimal_pattern_tmp = torch.stack(optimal_pattern_tmp)
        # optimal_pattern_tmp = optimal_pattern_tmp/torch.norm(optimal_pattern_tmp, dim=2, keepdim=True)
        
        # Check the shape of pattern vector
        if optimal_pattern_tmp.shape != (self.parallel_num, self.angle_num, self.angle_num, 1, 1):
            raise ValueError('Found pattern vector in wrong shape.')
        if detach == True:
            optimal_pattern_tmp = torch.clone(optimal_pattern_tmp.detach())
            
        if averageM == True:
            temp = torch.cat((optimal_pattern_tmp, optimal_pattern_tmp), dim=2)
            avg_response = temp[:, 0, :self.angle_num]
            for i in range(1, self.angle_num):
                avg_response = avg_response + temp[:, i, i:self.angle_num+i]
            avg_response = avg_response/self.angle_num
            # print(avg_response.shape)
            temp = torch.cat((avg_response, avg_response), dim=1)
            # print(temp.shape)
            for i in range(self.angle_num):
                # print(temp[:, i, self.angle_num-i:2*self.angle_num-i].shape)
                optimal_pattern_tmp[:, i] = temp[:, self.angle_num-i:2*self.angle_num-i]
                
        return optimal_pattern_tmp
    
    def get_p(self, x, dim, exp=False):
        if exp == True:
            x = torch.exp(x)
        return x/x.sum(dim=dim, keepdim=True)

    def pattern_loss(self, mode=4):
        if mode == 0:
            '''
            pattern regularisation, equivalent to diff_max_marginal(kernel, kernel).
            '''
            # mask = (1-torch.eye(self.angle_num)*2).unsqueeze(0)
            mask = torch.ones((self.parallel_num, self.angle_num, self.angle_num)) - torch.eye(self.angle_num).unsqueeze(0)
            mask = self.get_p(self.original_pattern, dim=2)[:, :, :, 0, 0]*mask
            return torch.exp((self.get_p((self.get_optimal_pattern(False)+1)/2.01, dim=2)[:, :, :, 0, 0]*mask).sum()/self.parallel_num)
        elif mode == 3:
            return self.get_p(torch.exp(self.get_optimal_pattern(False)+1), dim=2), self.get_p(torch.exp(self.original_pattern*2.01), dim=2)
        elif mode == 4:
            lock_direction = True
            threshold = 2
            
            base_num = int((self.angle_num/2+2)/4)
            forthPi = int(self.angle_num/4)
            halfPi = int(self.angle_num/2)
            base_stride = int(forthPi/base_num)
            
            
            rotational_cost = []
            for i in range(self.k_size**2):
                temp = []
                for j in range(self.parallel_num):
                    # print(self.angle_num, self.conv_w[j].shape, Generator_free(self.angle_num, self.k_size)(scale = self.k_size)[1:forthPi:base_stride].shape)
                    if lock_direction:
                        temp.append(self.norm_kernel(torch.cat((self.conv_w[j], self.conv_w[j].transpose(2,3).index_select(0, self.idx)), dim=0))*self.norm_kernel(Generator_free(self.angle_num, self.k_size)(scale = self.k_size)[1+i%halfPi:2*forthPi+i%halfPi:base_stride].clone()))
                    else:
                        temp.append(self.norm_kernel(self.conv_w[j])*self.norm_kernel(Generator_free(self.angle_num, self.k_size)(scale = self.k_size)[1+i%halfPi:forthPi+i%halfPi:base_stride].clone()))
                # print((torch.stack(temp).sum(dim=(-3, -2, -1)).std(1)))
                rotational_cost.append((torch.stack(temp).sum(dim=(-3, -2, -1)).std(1) - threshold).exp() - np.exp(-threshold))
            # print(rotational_cost[0].shape)
            rotational_cost = torch.cat(rotational_cost, dim=0)
            # print(rotational_cost.shape)
            return rotational_cost

    def update_optimal_pattern(self):
        '''
        Update the optimal_pattern parameter in a GC_node object.
        '''
        r = 0
        self.optimal_pattern = self.get_optimal_pattern(averageM=False)*(1-r)+self.optimal_pattern*r
        return
    
    def test_optimal_pattern(self, averageM=False):
        '''
        Always return the optimal patterm with additional Gab not packed.
        Shape: parallel_num x angle_num x pattern_vector(angle_num, 1, 1)
        '''
        kernels = self.pack_kernel(False)
        optimal_pattern_tmp = []
        for k in range(self.parallel_num):
            temp = kernels[k].view((self.angle_num, self.k_size**2))
            optimal_pattern_tmp.append(torch.matmul(temp, temp.T).view(self.angle_num, self.angle_num, 1, 1))
        optimal_pattern_tmp = torch.stack(optimal_pattern_tmp)
        # optimal_pattern_tmp = optimal_pattern_tmp/torch.norm(optimal_pattern_tmp, dim=2, keepdim=True)
        
        # Check the shape of pattern vector
        if optimal_pattern_tmp.shape != (self.parallel_num, self.angle_num, self.angle_num, 1, 1):
            raise ValueError('Found pattern vector in wrong shape.')
            
        if averageM == True:
            temp = torch.cat((optimal_pattern_tmp, optimal_pattern_tmp), dim=2)
            avg_response = temp[:, 0, :self.angle_num]
            for i in range(1, self.angle_num):
                avg_response = avg_response + temp[:, i, i:self.angle_num+i]
            avg_response = avg_response/self.angle_num
            # print(avg_response.shape)
            temp = torch.cat((avg_response, avg_response), dim=1)
            # print(temp.shape)
            for i in range(self.angle_num):
                # print(temp[:, i, self.angle_num-i:2*self.angle_num-i].shape)
                optimal_pattern_tmp[:, i] = temp[:, self.angle_num-i:2*self.angle_num-i]
        optimal_pattern_tmp = optimal_pattern_tmp.detach()
        
        
        return optimal_pattern_tmp

        
    def soft_max_marginal(self, xbs):
        '''
        Prenonmalised softmax but with non-normalised output.
        '''
        # xbs = (xbs+1)/2
        xbs = torch.exp(xbs)
        return xbs.sum(1, keepdim=True)
    
    def diff_max_marginal(self, xbs, l2_norm=True):
        '''
        Prenonmalised diff softmax but with non-normalised output.
        '''
        # xbs = (xbs+1)/2
        scale = -1/(self.angle_num)
        feature_map, index_map = xbs.max(1, keepdim=True)
        temp = (xbs!=feature_map)*scale + (xbs==feature_map)*3
        if l2_norm == True:
            xbs = torch.exp(xbs) * temp
        else:
            # xbs = torch.exp(xbs/xbs.max()) * temp
            xbs = torch.sigmoid(xbs) * temp
        return xbs.sum(1, keepdim=True), index_map
    
    def diff_cosine_marginal(self, x, parallel_i, exp=True):
        '''
        Compare the directional signal pattern to the optimal pattern of each
        direction in l2, use modified m as m*(I*c-b). 
        Return the most-matched direction.
        Return shape (batch_num, c, h, w)
        '''
        if exp:
            # x = torch.exp(torch.log(self.exp_degree[parallel_i])*x) - torch.exp(self.b[parallel_i])
            # m = torch.exp(torch.log(self.exp_degree[parallel_i])*self.optimal_pattern[parallel_i])
            # x = torch.exp(self.exp_degree[parallel_i]*x) - torch.exp(self.b[parallel_i])
            x = torch.exp(self.exp_degree[parallel_i]*x)
            m = torch.exp(self.exp_degree[parallel_i]*self.optimal_pattern[parallel_i])
            m = m - m.mean(dim=1, keepdim=True)*0.75
            norm_m = torch.norm(m, dim=1, keepdim=True)
            # m = m*(torch.eye(self.angle_num)*self.angle_num-1).view(self.angle_num, self.angle_num, 1, 1)
            # temp = torch.norm(x, dim=1, keepdim=True)
            # norm_m = torch.norm(m, dim=1, keepdim=True)
            # x_normed = x/(temp+0.000001*(temp==0))
            x_normed = x/torch.mean(norm_m)
            # x_normed = x/torch.mean(temp+0.00001*(temp==0))
            # x_norm = x
            m_normed = m/norm_m
        else:
            m = self.optimal_pattern[parallel_i]
            m = m - m.mean(dim=1, keepdim=True)*0.75
            norm_m = torch.norm(m, dim=1, keepdim=True)
            # m = m*(torch.eye(self.angle_num)*self.angle_num-1).view(self.angle_num, self.angle_num, 1, 1)
            # temp = torch.norm(x, dim=1, keepdim=True)
            # norm_m = torch.norm(m, dim=1, keepdim=True)
            # x_normed = x/(temp+0.000001*(temp==0))
            x_normed = x/torch.mean(norm_m)
            # x_normed = x/torch.mean(temp+0.00001*(temp==0))
            # x_norm = x
            m_normed = m/norm_m
        # print(m_normed.shape)
        # sdasd =asdasdasdas
        # print(x_norm.shape, m_norm.shape)
        temp = F.conv3d(input=x_normed.unsqueeze(1), weight=m_normed.unsqueeze(1))[:, :, 0]
        # return self.symmetric_max(temp)
        # print(temp.max(), temp.min())
        max_map, index_map = self.mono_max(temp, dim=1, keepdim=True)
        # print(max_map.max(), max_map.min())
        
        return max_map - torch.exp(self.b2[parallel_i]-7), index_map
    
    def symmetric_max(self, x):
        '''
        Gather output = X[argmax(X, |X|)]
        '''
        mask = x.abs().max(1, keepdim=True)[1]
        return x.gather(1, mask)
    
    def mono_max(self, x, dim=1, keepdim=True):
        '''
        Gather output = max(X)
        '''
        return x.max(dim, keepdim=keepdim) 
        
    def cosine_message_search(self, x, parallel_i, exp=True):
        '''
        Compare the directional signal pattern to the optimal pattern of each 
        direction in l2.
        Return the most-matched direction.
        Return shape (batch_num, c, h, w)
        '''
        if exp:
            # x = torch.exp(torch.log(self.exp_degree[parallel_i])*x) - torch.exp(self.b[parallel_i])
            # m = torch.exp(torch.log(self.exp_degree[parallel_i])*self.optimal_pattern[parallel_i])
            # x = torch.exp(self.exp_degree[parallel_i]*x) - torch.exp(self.b[parallel_i])
            x = torch.exp(self.exp_degree[parallel_i]*x)
            m = torch.exp(self.exp_degree[parallel_i]*self.optimal_pattern[parallel_i])
            # temp = torch.norm(x, dim=1, keepdim=True)
            norm_m = torch.norm(m, dim=1, keepdim=True)
            # x_normed = x/(temp+0.00001*(temp==0))
            x_normed = x/torch.mean(norm_m)
            # x_normed = x/torch.mean(temp+0.00001*(temp==0))
            # x_norm = x
            m_normed = m/norm_m
        else:
            m = self.optimal_pattern[parallel_i]
            # temp = torch.norm(x, dim=1, keepdim=True)
            norm_m = torch.norm(m, dim=1, keepdim=True)
            # x_normed = x/(temp+0.00001*(temp==0))
            x_normed = x/torch.mean(norm_m)
            # x_normed = x/torch.mean(temp+0.00001*(temp==0))
            # x_norm = x
            m_normed = m/norm_m
        
        
        # print(x_norm.shape, m_norm.shape)
        temp = F.conv3d(input=x_normed.unsqueeze(1), weight=m_normed.unsqueeze(1))[:, :, 0]
        # return self.symmetric_max(temp)
        # print(temp.max(), temp.min())
        max_map, index_map = self.mono_max(temp, dim=1, keepdim=True)
        
        return max_map - torch.exp(self.b2[parallel_i]), index_map
        # temp = F.leaky_relu(temp).max(1, keepdim=True)[0]
    
    def reorder_search(self, x, parallel_i):
        '''
        Reorder feature vector v, making strongest signal to be the first entry.
        Keeping the order of the sequence.
        Then combine the message in a learnable manner.
        '''
        b, c, h, w = x.shape
        if c != 8:
            raise ValueError('Channel number should be 1, but input has {} channels.'.format(c))
        index_map = self.mono_max(x, dim=1, keepdim=True)[1]
        temp = torch.arange(8, 0, -1).view((1, 8, 1, 1)).repeat(b, 1, h, w)
        temp = index_map - temp
        temp[temp<0] = temp[temp<0] + 8
        # print(x[0, :, 1, 1])
        x = torch.gather(x, 1, temp)
        # print(x[0, :, 1, 1])
        return self.reorder_combine[parallel_i](x), index_map
        
    def stretch_message_search(self, x, parallel_i):
        '''
        sqrt(sum((r - v)^2)l)
        '''
        c = torch.sqrt(torch.tensor(self.angle_num/2))
        x = x.unsqueeze(2).repeat(1, 1, self.angle_num, 1, 1)
        x = c - torch.norm(x - self.optimal_pattern[parallel_i].unsqueeze(0), dim=2, keepdim=False)
        # return self.symmetric_max(temp)
        return self.mono_max(x)
    
    def forward(self, xb, l2_norm=True):
        # b, c, h, w = xb.shape
        self.update_optimal_pattern()
        # Must be nature number, The smaller the stonger. Minimum = 1.
        denoise_rate = 1
        temp_parallel = []
        
        kernel = self.pack_kernel()
        if l2_norm == True:
            for j in range(self.parallel_num):
                l2 = self.l2_layer(xb[:, j].unsqueeze(1), self.k_size)
                # print(xb[:, j].view(b, 1, h, w).shape)
                temp_parallel.append(F.conv2d(xb[:, j].unsqueeze(1), kernel[j], bias=None, stride=1, padding=int(self.padding), groups=self.in_groups)/(l2+0.0001))
        else:
            for j in range(self.parallel_num):
                temp_parallel.append(F.conv2d(xb[:, j].unsqueeze(1), kernel[j], bias=None, stride=1, padding=int(self.padding), groups=self.in_groups))
        
        if l2_norm==True and np.random.randint(0, 10)<1:
            # if temp_parallel[0][0].max() > 1.1+self.mac_b[0]:
            if temp_parallel[0][0].max() > 1.0:
                print(temp_parallel[0][0].max())
                raise ValueError("Temp maximum exeed 1 issue.")
                
        data = []
        index_maps = []
        for j in range(self.parallel_num):
            # temp = self.bn(temp_parallel[j])
            if self.messageIntegration == 'diffMax':
                feature_map, index_map = self.diff_max_marginal(temp_parallel[j], l2_norm=l2_norm)
            elif self.messageIntegration == 'reorder':
                feature_map, index_map = self.reorder_search(temp_parallel[j], j)
            elif self.messageIntegration == 'cosine':
                feature_map, index_map = self.cosine_message_search(temp_parallel[j], j)
            elif self.messageIntegration == 'diffCosine':
                feature_map, index_map = self.diff_cosine_marginal(temp_parallel[j], j)
            elif self.messageIntegration == 'stretch':
                feature_map, index_map = self.stretch_message_search(temp_parallel[j], j)
            elif self.messageIntegration == 'monoMax':
                feature_map, index_map = self.monoMax(temp_parallel[j])
            if self.messageIntegration == 'softMax':
                feature_map = self.soft_max_marginal(temp_parallel[j])
                _, index_map = self.monoMax(temp_parallel[j])
            data.append(feature_map)
            index_maps.append(index_map)
            # data.append(F.softmax(temp_parallel[j]).shape)
        # print(data[0].shape)
        # print(torch.max(torch.cat(data, dim=1).detach()), torch.max(torch.cat(temp_parallel, dim=1).detach()), torch.max(torch.tensor([iw for iw in self.mac_b])).detach())
        return torch.cat(data, dim=1), torch.cat(index_maps, dim=1)


class GC_Block(nn.Module):
    def __init__(self, in_num, mid_num, out_num, angle, k_size=9, inScaleLevel=0, scaleLevel=(1, 2, 3, 4), parallel_num=1, p2nd_k=3, share=False, identical=True):
        super().__init__()
        self.in_num = in_num
        self.mid_num = mid_num
        self.out_num = out_num
        self.angle = angle
        self.inScaleLevel = inScaleLevel
        self.scaleLevel = scaleLevel
        self.parallel_num = parallel_num
        self.scale_num = len(scaleLevel)
        self.k_size = k_size
        self.identical = identical
        
        self.bottleNeck = nn.Conv2d(self.in_num, self.mid_num, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.pre_GConv = nn.Conv2d(self.mid_num, parallel_num, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.resi = nn.Conv2d(self.mid_num, self.mid_num*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.share = share
        if self.share == False:
            self.GConv = GabLayer_DoG(self.angle, self.k_size, parallel_num=parallel_num, identical=identical)
        # self.combineGC = nn.Conv2d(self.parallel_num, self.out_num, kernel_size=1, stride=1, padding=0)
        self.combine = nn.Conv2d((0*in_num+3*self.mid_num+0*self.out_num+2*self.parallel_num)*1+self.parallel_num*0, self.out_num, kernel_size=1, stride=1, padding=0)
        
        
        self.bn_bottleNeck = nn.BatchNorm2d(self.mid_num)
        self.bn_gc = nn.BatchNorm2d(self.parallel_num)
        self.bn_resi = nn.BatchNorm2d(self.mid_num*2)
        self.bn_combine = nn.BatchNorm2d(self.out_num)
        
        # self.indexFeaturize = indexFeaturize(self.angle, self.parallel_num)
        # self.indexFeaturize = indexFeaturize_relative(self.angle, self.parallel_num)
        
        # self.pre_pooling2nd_combine = nn.Conv2d((0*in_num+3*self.mid_num+0*self.out_num+1*self.parallel_num)*1, 4, kernel_size=1, stride=1, padding=0)
        # self.pooling_2nd = Pooling2nd(p2nd_k, dilation=2, stride=1, base_mode='selective', corr_mode='plain')
        
        self.exp_message1 = exp_message()
        self.exp_message2 = exp_message()
        
    def l2_layer(self, xb, kernel_size=3):
        '''
        Only on spatial not on depth.
        '''
        xb_padded = F.pad(xb, pad=(kernel_size//2, kernel_size//2)*2, mode='reflect')
        return F.lp_pool2d(xb_padded, norm_type=2, kernel_size=kernel_size, stride=1)
    
    def soft_max_marginal(self, x):
        '''
        Prenonmalised softmax but with non-normalised output.
        '''
        # return torch.exp(x).sum(1, keepdim=True)
        # output = []
        # for i in range(self.parallel_num):
        #     temp = x[:, i::self.parallel_num]
        #     output.append(temp.max(1, True)[0])
        # return torch.cat(output, dim=1)
            
        scale = 1/(self.angle)
        output = []
        for i in range(self.parallel_num):
            temp = x[:, i::self.parallel_num]
            mask = (temp!=temp.max(1, keepdim=True)[0])*scale + (temp==temp.max(1, keepdim=True)[0])*1
            output.append((torch.exp(temp) * mask).sum(1, keepdim=True))
        return torch.cat(output, dim=1)
    
    def symmetric_max(self, x, dim=1, keepdim=True):
        '''
        return max(max, |min|)*original sign.
        '''
        output = []
        for i in range(self.parallel_num):
            temp = x[:, i::self.parallel_num]
            output.append(temp.gather(1, temp.abs().max(1, keepdim=True)[1]))
        return torch.cat(output, dim=1)
    
    def max_channel(self, x, all_directional_indeces):
        output = []
        output_index = []
        for i in range(self.parallel_num):
            temp = x[:, i::self.parallel_num]
            temp_index = all_directional_indeces[:, i::self.parallel_num]
            
            feature, scale_index = temp.max(1, keepdim=True)
            directional_index = temp_index.gather(1, scale_index)
            
            output.append(feature)
            output_index.append(directional_index)
        
        output = torch.cat(output, dim=1)
        output_index = torch.cat(output_index, dim=1)
        # return output, output_index.gather(1, output.max(1, keepdim=True)[1])
        return output, output_index, scale_index
    
    def pattern_loss(self, mode=4):
        '''
        Refer to GConv node pattern_loss.
        Return 0 in GConv-sharing mode. 
        '''
        if self.share == True:
            return 0
        else:
            # print(self.GConv.pattern_loss())
            return self.GConv.pattern_loss(mode).mean()
    
    def forward(self, x, sharing_GConv=None, l2_norm=True):
        if self.share == True and sharing_GConv == None:
            ValueError('In share model, must pass a GConv node into the block.')
        x1 = self.bn_bottleNeck(F.leaky_relu(self.bottleNeck(x)))
        resi_x = self.bn_resi(F.leaky_relu(self.resi(x1)))
        
        temp = self.pre_GConv(x1)
        gc = []
        gc_index = []
        for i in range(len(self.scaleLevel)):
            if self.share == False:
                feature_map, index_maps = self.GConv(F.max_pool2d(F.leaky_relu(temp), kernel_size=self.scaleLevel[i], stride=self.scaleLevel[i]), l2_norm=l2_norm)
                gc.append(F.interpolate(feature_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear'))
                gc_index.append(nn.functional.interpolate(index_maps*1.0, scale_factor=self.scaleLevel[i], mode='nearest'))
            if self.share == True:
                feature_map, index_maps = sharing_GConv(F.max_pool2d(F.leaky_relu(temp), kernel_size=self.scaleLevel[i], stride=self.scaleLevel[i]), l2_norm=l2_norm)
                gc.append(F.interpolate(feature_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear'))
                gc_index.append(nn.functional.interpolate(index_maps*1.0, scale_factor=self.scaleLevel[i], mode='nearest'))
        gc = torch.cat(gc, dim=1)
        gc_index = torch.cat(gc_index, dim=1)
        # print(temp.detach().max(), gc.detach().max(), gc.detach().min())
        # gc = self.soft_max_marginal(gc)
        gc, gc_index, scale_index = self.max_channel(gc, gc_index)
        # print(gc.shape, gc_index.shape)
        if l2_norm == True:
            # gc2 = self.bn_gc(gc*self.l2_layer(gc))
            # gc2 = gc*self.l2_layer(temp)
            # print(gc.max(), gc.min())
            
            # gc2 = torch.exp(torch.log(gc)*self.l2_layer(temp))
            # gc2 = torch.exp(torch.log(torch.sigmoid(gc)*2)*self.l2_layer(temp))
            gc2 = []
            for i in range(self.parallel_num):
                gc2.append(self.exp_message1(self.l2_layer(temp[:, i].unsqueeze(1)), gc[:, i].unsqueeze(1)))
            gc2 = torch.cat(gc2, dim=1)
        else:
            # gc2 = self.bn_gc(gc)
            # gc2 = gc*self.l2_layer(temp, kernel_size=self.k_size)
            
            # gc2 = torch.exp(torch.log(gc)*self.l2_layer(temp, kernel_size=self.k_size))
            # gc2 = torch.exp(torch.log(torch.sigmoid(gc)*2)*self.l2_layer(temp, kernel_size=self.k_size))
            gc2 = self.exp_message1(self.l2_layer(temp, kernel_size=self.k_size), gc)
        # gc = self.symmetric_max(gc)
        # gc_com = self.combineGC(gc)
        b, c, h, w = gc2.shape
        # print(gc2.view(b, c, -1).min(dim=2)[0].shape)
        x2 = torch.cat((x1, resi_x, gc2, gc), dim=1)
        '''
        Second_order pooling
        '''
        # temp = F.gelu(self.pre_pooling2nd_combine(x2))
        # x2 = torch.cat((x2, self.exp_message(x2, F.relu(self.pooling_2nd(x2, gc_index)))), dim=1)
        # temp = self.exp_message2(temp, self.pooling_2nd(temp, mask=gc_index))
        
        '''
        Index featurisation
        '''
        # temp = self.indexFeaturize(gc_index, gc, self.scaleLevel)
        # print(temp.max().detach(), temp.min().detach())
        # x2 = x2 + temp
        # x2 = torch.cat((x2, temp), dim=1)
        x2 = self.bn_combine(F.leaky_relu(self.combine(x2)))
        # print(x2.min())
        return x2, gc2[:, :1], gc.sum(dim=1, keepdim=True)/self.parallel_num, (gc_index/(self.angle-1))*255, (scale_index/(len(self.scaleLevel)-1))*255
    
#class
class GConv(nn.Module):
    def __init__(self, angle, parallel_num, k_size, identical=True):
        super().__init__()
#        self.max1x1 = nn.MaxPool3d((8, 1, 1), stride=(1, 1, 1))
        self.angle = angle
        self.parallel_num = parallel_num
        
        self.l1 = nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.bn_l1 = nn.BatchNorm2d(4)
        self.l2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn_l2 = nn.BatchNorm2d(8)
        # self.l3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.l3 = GC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.bn_l3 = nn.BatchNorm2d(8)
        
        self.combine1 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.bn_combine1 = nn.BatchNorm2d(8)
        
        self.combine2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.bn_combine2 = nn.BatchNorm2d(8)
        
        
        self.GC_Block1 = GC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.GC_Block2 = GC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.GC_Block3 = GC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        
        # self.p2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.p2 = GC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.bn_p2 = nn.BatchNorm2d(8)
        self.p4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
    
    def pack_pattern_loss(self, mode):
        '''
        Pack regularisation loss term on all non-sharing GConv nodes.
        '''
        # GC_blocks = torch.stack([self.l3.pattern_loss(), self.GC_Block1.pattern_loss(), self.GC_Block2.pattern_loss(), self.GC_Block3.pattern_loss(), self.p2.pattern_loss()])
        if mode == 3:
            GC_blocks = [self.l3.pattern_loss(mode), self.GC_Block1.pattern_loss(mode), self.GC_Block2.pattern_loss(mode), self.GC_Block3.pattern_loss(mode), self.p2.pattern_loss(mode)]
            
            op = []
            ori = []
            for i, j in GC_blocks:
                op.append(i[:, :, : ,0, 0])
                ori.append(j[:, :, : ,0, 0])
            return torch.stack(op), torch.stack(ori)
            # return torch.sum(GC_blocks)/len(GC_blocks)
        elif mode == 4:
            return torch.mean(torch.stack((self.l3.pattern_loss(mode), 
                                           self.GC_Block1.pattern_loss(mode),
                                           self.GC_Block2.pattern_loss(mode),
                                           self.GC_Block3.pattern_loss(mode),
                                           self.p2.pattern_loss(mode))))

    def forward(self, x, l2_norm=True, return_position=-1):
        x = self.bn_l1(F.relu(self.l1(x)))
        x = self.bn_l2(F.relu(self.l2(x)))
        x0 = self.l3(x, l2_norm=l2_norm)
        x = self.bn_l3(F.relu(x0[0]))
        
        x1 = self.GC_Block1(x, l2_norm=l2_norm)
        x2 = self.GC_Block2(x1[0], l2_norm=l2_norm)
        
        
        
        temp = torch.cat((x1[0], x2[0]), dim=1)
        temp = self.bn_combine1(F.leaky_relu(self.combine1(temp)))
        
        x3 = self.GC_Block3(temp, l2_norm=l2_norm)
        
        temp = torch.cat((x, x3[0]), dim=1)
        temp = self.bn_combine2(F.leaky_relu(self.combine2(temp)))
        
        x4 = self.p2(temp, l2_norm=l2_norm)
        fc = self.bn_p2(F.relu(x4[0]))
        fc = torch.sigmoid(self.p4(fc))
        
        target_map = [torch.sigmoid(x0[2]), torch.sigmoid(x1[2]), torch.sigmoid(x2[2]), torch.sigmoid(x3[2]), torch.sigmoid(x4[2]), fc]
        index_maps = [x1[3], x2[3], x3[3]]
        scale_index_maps = [x1[4], x2[4], x3[4]]
        # print(index_maps[1].shape)
        # print(target_map[0].max(), target_map[0].min())
        return target_map[return_position], x1[1], x2[1], x3[1], index_maps, scale_index_maps
        # return fc, x1[1], x2[1], x3[1]


def BCE_extention(loss_func, model):
    return loss_func + model.pack_pattern_loss()
    
    
def fit(model, epochs, loss_func, mask_loss, lambda_, opt, trainset, testset, folder, testimg, warm_up=False, l2_norm=True, resume=False):
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, min_lr=0.00001, patience=11, cooldown=5, verbose=True)
    loss_recorder = 1
    if resume:
        loss_log = list(np.load(folder + '/log.npy'))
    else:
        loss_log = list()

    model_recorder = deepcopy(model)
    warm_up_epochs = 10
    steerable_cost_involveEpoch = 2500
    block_num = 5
    lambda_loss = lambda_
    loss_mode = 4
    
    if warm_up == True:
        if l2_norm == False:
            raise ValueError('Can only pretrain under l2_norm mode.')
    print('lambda: {}'.format(lambda_loss))
    # default_position
    position = -1
    for epoch in range(len(loss_log), epochs):
        time1 = time.time()
        model.train()
        for xb, yb in trainset:
            opt.zero_grad()
#            print(xb.shape, model(xb, l2_norm)[0].shape, yb.shape)
#            yb_downsample = F.max_pool2d(yb, kernel_size = 2, stride=2)
#            print(torch.stack((yb, yb)))
            if warm_up==True:
                position = warmup_position(epoch, warm_up_epochs=warm_up_epochs, block_num=block_num)
                loss = loss_func(model(xb, l2_norm, position)[0], yb)
                if position >= block_num:
                    warm_up = False
            else:
                loss = loss_func(model(xb, l2_norm)[0], yb)
            
            if lambda_loss > 0 and epoch > steerable_cost_involveEpoch:
                if loss_mode == 3:
                    op, ori = model.pack_pattern_loss(loss_mode)
                    loss = loss + lambda_loss*mask_loss(op, ori)
                elif loss_mode == 4:
                    speed_control = 1-(1.05**(steerable_cost_involveEpoch-epoch))*(epoch > steerable_cost_involveEpoch)
                    loss = loss + speed_control*lambda_loss*model.pack_pattern_loss(loss_mode)
                # loss = loss + lambda_loss*model.pack_pattern_loss()
#            loss = loss_func(torch.stack(model(xb, l2_norm)[:2]), torch.stack((yb, yb)))
#            loss_mask = mask_loss(torch.stack(model(xb, l2_norm)[1:]), torch.stack((yb_downsample, yb_downsample, yb_downsample)))
            loss.backward()
#            loss_mask.backward()
            opt.step()
        model.eval()
        
        
        with torch.no_grad():
            if epoch%5 == 0:
                os.system("mkdir "+folder+"/ep"+str(epoch))
                torch.save(model, folder + '/model')
                # print('position: '+str(position))
                a = model(testimg, l2_norm, position)
                index = 0
                for i in a[1:4]:
                    pmap = to_array(i)[0, 0]
                    pmap = pmap-pmap.min()
                    pmap = 254*(pmap/pmap.max())
                    cv2.imwrite(folder+"/ep"+str(epoch)+'/pmap'+str(index)+'_ep'+str(epoch)+'.png', pmap)
                    index += 1
                index = 0
                for i in a[4]:
                    cv2.imwrite(folder+"/ep"+str(epoch)+'/index'+str(index)+'_ep'+str(epoch)+'.png', to_array(i[0,0]))
                    index += 1
                for i in a[5]:
                    cv2.imwrite(folder+"/ep"+str(epoch)+'/scale_index'+str(index)+'_ep'+str(epoch)+'.png', to_array(i[0,0]))
                    index += 1
                cv2.imwrite(folder+"/ep"+str(epoch)+'/out1_ep'+str(epoch)+'.png', to_array(a[0][0,0])*255) 
                np.save(folder + '/log', loss_log)
                # cv2.imwrite(folder+"/ep"+str(epoch)+'/out2_ep'+str(epoch)+'.png', to_array(a[0][0,1])*255)
                # cv2.imwrite(folder+"/ep"+str(epoch)+'/out3_ep'+str(epoch)+'.png', to_array(a[0][0,2])*255)
            loss_test = [loss_func(model(xb, l2_norm)[0], yb) for xb, yb in testset]
            loss_test = np.average(loss_test)
            loss_log.append((loss_test, model.pack_pattern_loss(loss_mode)))
            if loss_test<loss_recorder:
                model_recorder = deepcopy(model)
            lr_schedular.step(loss_test)
            # print('epoch:{}\tloss:{:.5f}\t{}*loss_ext:{:.5f}\ttime:{:.5f}'.format(epoch, loss_test, lambda_loss, lambda_loss*to_array( model.pack_pattern_loss()), time.time()-time1))
            print('epoch:{}\tloss:{:.5f}\t{}*loss_ext:{:.5f}\ttime:{:.5f}'.format(epoch, loss_test, lambda_loss, to_array(model.pack_pattern_loss(loss_mode)), time.time()-time1))
        if len(loss_log)>20 and (np.max(loss_log[-30:])-np.min(loss_log[-30:]))<0.0005:
            break
    return loss_log, model_recorder

def warmup_position(epoch, warm_up_epochs=10, block_num=3):
    '''
    In the case of warm-up training, determine the aiming block index.
    '''
    position = int(epoch//warm_up_epochs)
    if position >= block_num:
        position == block_num
    return position
    
    
def activate(a):
    return 500*(torch.sigmoid(a)*(1-torch.sigmoid(a))-0.05)

def leaky_activate(a):
    return torch.sigmoid(500*(torch.sigmoid(a)*(1-torch.sigmoid(a))-0.05))*1.1-0.1

class Generator(nn.Module):
    def __init__(self, num, kernel_size, normalise=False):
        '''
        x_sigma >= k/2
        y_sigma <= k/18
        gamma = x_sigma/y_sigma
        gamma >= 9
        '''
        super(Generator, self).__init__()
        self.num = num
        self.kernel_size = torch.tensor(kernel_size)
        self.lambda_ = torch.nn.Parameter(2*self.kernel_size.float())
        self.theta = torch.nn.Parameter((torch.arange(0, 1, 1.0 / float(self.num))* np.pi).view(self.num, 1, 1))
        self.theta.requires_grad = False
        # self.lambda_.requires_grad = False
        self.phi = torch.nn.Parameter(torch.tensor(0.0))

        self.sigma = torch.nn.Parameter((1/2+torch.rand(1)/8)*self.kernel_size)
        # self.gamma = torch.nn.Parameter(8.0+torch.rand(1)*8.0)
        # self.gamma = torch.nn.Parameter(3+18*torch.rand(1))
        self.gamma = torch.nn.Parameter(12+9*torch.rand(1))
        self.amplitude = torch.nn.Parameter(1+3*torch.abs(torch.randn(1)))
        self.normalise = normalise

    def forward(self, scale = None):
        if scale > self.kernel_size:
            raise ValueError('In current scheme, only support kernel down sampling.')
            
        x = Variable(torch.arange(-0.9, 1.1, (1.0 / float(scale)) * 2.0))
        y = Variable(torch.arange(-0.9, 1.1, (1.0 / float(scale)) * 2.0))

        x = x.view(1, -1).repeat(self.num, scale, 1)
        y = y.view(-1, 1).repeat(self.num, 1, scale)

        x1 = (x * torch.cos(self.theta)) + (y * torch.sin(self.theta))
        y1 = (-x * torch.sin(self.theta)) + (y * torch.cos(self.theta))

        sigma_y = self.sigma/self.gamma
        g0 = torch.exp(-0.5 * ((x1**2 / (self.sigma)**2) + (y1**2 / sigma_y**2)))
        g1 = torch.cos((2.0 * np.pi * (x1 / (self.lambda_*scale/self.kernel_size))) + self.phi*0)
        g_real = (g0 * g1) * self.amplitude

        if self.normalise:
            g_mean = g_real.mean(dim=(1,2))
            g_mean_rep = g_mean.view(self.num, 1, 1).repeat(1, scale, scale)
            g_real = g_real-g_mean_rep
            g_real = g_real/torch.norm(g_real, dim=(1,2)).view(self.num, 1, 1).repeat(1, scale, scale)
        return g_real.view(self.num, 1, scale, scale)
    
class Generator_free(nn.Module):
    def __init__(self, num, kernel_size, normalise=False):
        '''
        x_sigma >= k/2
        y_sigma <= k/18
        gamma = x_sigma/y_sigma
        gamma >= 9
        '''
        super(Generator_free, self).__init__()
        self.num = num
        self.kernel_size = torch.tensor(kernel_size)
        self.lambda_ = torch.nn.Parameter(2*self.kernel_size.float())
        self.theta = torch.nn.Parameter((torch.arange(0, 1, 1.0 / float(self.num))* np.pi).view(self.num, 1, 1))
        self.theta.requires_grad = False
        # self.lambda_.requires_grad = False
        self.phi = torch.nn.Parameter((np.pi/6+torch.rand(1)*np.pi/6)*(-1+2*(torch.rand(1)>0.5)))

        self.sigma = torch.nn.Parameter((1/2+torch.rand(1)/8)*self.kernel_size/2)
        # self.gamma = torch.nn.Parameter(8.0+torch.rand(1)*8.0)
        self.gamma = torch.nn.Parameter(3+12*torch.rand(1))
        # self.gamma = torch.nn.Parameter(12+9*torch.rand(1))
        self.amplitude = torch.nn.Parameter(1+3*torch.abs(torch.randn(1)))
        self.normalise = normalise

    def forward(self, scale = None):
        if scale > self.kernel_size:
            raise ValueError('In current scheme, only support kernel down sampling.')
            
        x = Variable(torch.arange(-0.9, 1.1, (1.0 / float(scale)) * 2.0))
        y = Variable(torch.arange(-0.9, 1.1, (1.0 / float(scale)) * 2.0))

        x = x.view(1, -1).repeat(self.num, scale, 1)
        y = y.view(-1, 1).repeat(self.num, 1, scale)

        x1 = (x * torch.cos(self.theta)) + (y * torch.sin(self.theta))
        y1 = (-x * torch.sin(self.theta)) + (y * torch.cos(self.theta))

        sigma_y = self.sigma/self.gamma
        g0 = torch.exp(-0.5 * ((x1**2 / (self.sigma)**2) + (y1**2 / sigma_y**2)))
        g1 = torch.cos((2.0 * np.pi * (x1 / (self.lambda_*scale/self.kernel_size))) + self.phi)
        g_real = (g0 * g1) * self.amplitude

        if self.normalise:
            g_mean = g_real.mean(dim=(1,2))
            g_mean_rep = g_mean.view(self.num, 1, 1).repeat(1, scale, scale)
            g_real = g_real-g_mean_rep
            g_real = g_real/torch.norm(g_real, dim=(1,2)).view(self.num, 1, 1).repeat(1, scale, scale)
        return g_real.view(self.num, 1, scale, scale)
    
def preprocess(x, y):
#    print(x.shape, y.shape)
    return x.transpose(1, 3).transpose(2, 3).view(-1, 3, 48, 48), y.view(-1, 1, 48, 48)
    # return x.transpose(1, 3).transpose(2, 3).view(-1, 3, 48, 48), y.transpose(1, 3).transpose(2, 3).view(-1, 1, 48, 48)
    # return x.transpose(1, 3).transpose(2, 3).view(-1, 3, 48, 48), y.transpose(1, 3).transpose(2, 3).view(-1, 1, 48, 48)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl= dl
        self.func = func
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def get_data(train_dataset, batch_size, test_dataset=None):
    if test_dataset is not None:
        return(DataLoader(train_dataset, batch_size=batch_size), 
               DataLoader(test_dataset, batch_size=batch_size*2))
    return DataLoader(train_dataset, batch_size=batch_size)

#def illustration_simple(img_path, file_name, model):
#    img = torch.from_numpy(cv2.imread(img_path, 0))
#    img = img.view(1,1,*img.shape).float()
#    imwrite(model(img, l2_norm), file_name)

def get_filenames(path):
    files = os.listdir(path)
    files.sort()
    return files

def to_array(tensor):
    return tensor.detach().numpy()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

cwd = os.getcwd()

# standard data
#img_path = '/home/mbaxszlh/OneDrive/workspace/dataset/standard/axon/5000img/img/'
#label_path = '/home/mbaxszlh/OneDrive/workspace/dataset/standard/axon/5000img/label/'

# standard_sk data


# def main(folder, x_data, y_data, angle, epochs=300):
#     print(folder)
#     os.system('mkdir '+folder)
#     os.system("mkdir "+folder+"/test")
#     os.system("mkdir "+folder+"/test2")
#     os.system("mkdir "+folder+"/test3")
#     batch_size = 32
#     data = TensorDataset(x_data, y_data)
#     volume = len(data)
#     train_data, test_data = torch.utils.data.dataset.random_split(data, (round(volume*0.8), volume-round(volume*0.8)))
#     print('training volume: ', volume)
#     train_data = WrappedDataLoader(get_data(train_data, batch_size), preprocess)
#     test_data = WrappedDataLoader(get_data(test_data, batch_size), preprocess)
    
#     model = GConv(angle)
#     if torch.cuda.is_available():
#         print('cuda')
#         model = model.cuda()
#     print('model parameters: ', get_n_params(model))
# #    print(model.GCBlock4.Gab.amplitude)
#     print(model.GCBlock4.conv_w[0][0])
#     loss_func = nn.BCELoss()
#     loss_func2 = nn.BCELoss()
#     opt = torch.optim.Adam(model.parameters(), lr = 0.01)
    
#     img = cv2.imread('s1_051.tif')[:548, :960]
#     # img = cv2.resize(img,None,fx=0.5,fy=0.5)
#     img = torch.from_numpy(img)
#     img = img.transpose(0, 2).transpose(1, 2)
#     img = img.view(1, *img.shape).float()
#     #dic = np.load('dataset/movie0_236_cor.npy').item()
#     #coor = dic['coordinates']
#     #conct = dic['connections']
    
#     #train
#     loss_recorder, model = fit(model, epochs, loss_func, loss_func2, opt, train_data ,test_data, folder, img)
# #    print(model.GCBlock4.Gab.amplitude)
#     print(model.GCBlock4.conv_w[0][0])
#     torch.save(model, folder + '/model')
#     np.save(folder + '/log', loss_recorder)
    
#     a = model(img, l2_norm)
#     index = 0
#     for i in a[1:]:
#         pmap = to_array(i)[0, 0]
#         pmap = 255*(pmap/pmap.max())
#         cv2.imwrite(folder + '/pmap'+str(index)+'.png', pmap)
#         index += 1
#     cv2.imwrite(folder + '/out1.png', to_array(a[0][0,0])*255) 
#     # cv2.imwrite(folder + '/out2.png', to_array(a[0][0,1])*255) 
#     # cv2.imwrite(folder + '/out3.png', to_array(a[0][0,2])*255) 
#     #
# #    model = torch.load('model')
#     ##loss_func = nn.BCELoss()
#     ##opt = torch.optim.Adam(model.parameters(), lr = 0.001)
#     ##train
#     ##loss_recorder, model = fit(model, 100, loss_func, opt, train_data ,test_data)
#     test_list = os.listdir('test_imgs/')
    
#     for i in test_list:
#         print(i)
#         img = cv2.imread('test_imgs/'+i)[:548, :960]
#         # img = cv2.resize(img,None,fx=0.5,fy=0.5)
#         img = torch.from_numpy(img)
#         img = img.transpose(0, 2).transpose(1, 2)
#         img = img.view(1, *img.shape).float()
#         temp = model(img, l2_norm)[0][0]*255
#         cv2.imwrite(folder+'/test/'+i[:-3]+'png', to_array(temp[0]))
#         # cv2.imwrite(folder+'/test2/'+i[:-3]+'png', to_array(temp[1]))
#         # cv2.imwrite(folder+'/test3/'+i[:-3]+'png', to_array(temp[2]))
#     return model
 
def code_testimg(tensor):
    '''
    Resize test image to match the processing rescaling factor.
    '''
    _, _, ori_h, ori_w = tensor.shape
    h = int(ori_h/12)*12
    w = int(ori_w/12)*12
    return F.interpolate(tensor, size=(h, w), mode='bilinear'), ori_h, ori_w

def uncode_testimg(tensor, ori_h, ori_w):
    '''
    The inverse funtion of code_testing.
    '''
    return F.interpolate(tensor, size=(ori_h, ori_w), mode='bilinear')
    

def test_model(model, img, filename, l2_norm):
    #single image
    #img = cv2.resize(img,None,fx=0.5,fy=0.5)
    img = torch.from_numpy(img)
    img = img.transpose(0, 2).transpose(1, 2)
    img = img.view(1, *img.shape).float()
    a = model(img, l2_norm)
    #recommand using png, as tif may cause unexpected output
    cv2.imwrite(filename, to_array(a[0][0,0])*255)
    
def apply_model(model, l2_norm, in_folder, out_folder, video_tag=None):
#    segmentation for video
    kernel = np.ones((3,3))
    files = np.sort(os.listdir(in_folder))
    index = 0
    os.system('rm -r '+out_folder)
    os.system('mkdir '+out_folder)
    os.system('mkdir '+out_folder+'axon')
    os.system('mkdir '+out_folder+'blob')
    os.system('mkdir '+out_folder+'cell')
    os.system('mkdir '+out_folder+'com')
    
    if files[0] == '.DS_Store':
        files = files[1:]
    for i in files:
        
        img = cv2.imread(in_folder+i)[:672, :960]
        com = np.copy(img)
        #img = cv2.resize(img,None,fx=0.5,fy=0.5)
        img = torch.from_numpy(img)
        img = img.transpose(0, 2).transpose(1, 2)
        img = img.view(1, *img.shape).float()
        a = model(img, l2_norm)[0]
        temp = str(index)
        temp = temp.zfill(3)
        print(out_folder+video_tag+temp+'.png')
#        print(a[0][0,0].shape)
        axon = to_array(a[0, 0])*255
        cell = cv2.dilate(to_array(a[0, 2])*255, kernel)
        blob = cv2.dilate(to_array(a[0, 1])*255, kernel)
        com[axon>100] = 255
        com[cell>100, 1] = 255
        com[blob>100, 2] = 255
        
        cv2.imwrite(out_folder+'axon/'+video_tag+temp+'.png', axon)
        cv2.imwrite(out_folder+'blob/'+video_tag+temp+'.png', blob)
        cv2.imwrite(out_folder+'cell/'+video_tag+temp+'.png', cell)
        cv2.imwrite(out_folder+'com/'+video_tag+temp+'.png', com)
        index += 1
        
def visualise_com(model, in_folder, out_folder, video_tag=None):
#    segmentation for video
    kernel = np.ones((3,3))
    files = np.sort(os.listdir(in_folder))
    index = 0

    if files[0] == '.DS_Store':
        files = files[1:]
    for i in range(len(files)):
        temp = str(index)
        temp = temp.zfill(3)
        img = cv2.imread(in_folder+files[i])[:672, :960]
        com = np.copy(img)
        axon = cv2.imread(out_folder+'axon/'+video_tag+temp+'.png', 0)
        blob = cv2.imread(out_folder+'blob/'+video_tag+temp+'.png', 0)
        cell = cv2.imread(out_folder+'cell/'+video_tag+temp+'.png', 0)
        axon = (cv2.dilate(axon, kernel, iterations = 2)>10)*1
        axon = skeletonize(axon.astype('uint8'))*255
        com[axon>100] = 255
        com[cell>100, 1] = 255
        com[blob>100, 2] = 255
#        asdasd =asdasdad
        cv2.imwrite(out_folder+'com2/'+video_tag+temp+'.png', com)
        index += 1   

def correlation_visual(model):
    pattern = to_array(model.GC_Block3.GConv.test_optimal_pattern()[:, :, :, 0, 0])
    n, theta, _ = pattern.shape
    fig = plt.figure()
    gs = fig.add_gridspec(n+1, n)
    fig.set_size_inches(1.67*n*theta/8, 2.5*(n+1)+(theta-8)/2, forward=True)
    axs = []
    fig.suptitle('directional pattern correlation')
    for p in range(n):
        axs.append(fig.add_subplot(gs[p, :]))
        for i in range(len(pattern[p])):
            axs[-1].plot(pattern[p][i], label=i)
        if p == n-1:
            axs[-1].set_xlabel('direction index')
        axs[-1].set_ylabel('correlation {}'.format(p+1))
        if p == 0:
            axs[-1].legend(bbox_to_anchor=(1.05, 1))
    for p in range(n):
        axs.append(fig.add_subplot(gs[n, p]))
        axs[-1].imshow(pattern[p])
        axs[-1].set_xlabel('correlation {}'.format(p+1))
# correlation_visual(temp) 

def convW_visual(model):
    kernels = model.GC_Block3.GConv.conv_w
    n = len(kernels)
    d, _, _, k_size = kernels[0].shape
    fig, axs = plt.subplots(n, d+1)
    fig.set_size_inches(2.5*(d+1), 2.7*n, forward=True)
    for i in range(n):
        axs[i, 0].imshow(to_array(model.GC_Block3.GConv.Gab[i](k_size)[1, 0]))
        axs[i, 0].title.set_text('Gabor(k={})'.format(i))
        for j in range(0, d):
            axs[i, j+1].imshow(to_array(kernels[i][j, 0]))
            axs[i, j+1].title.set_text('k={} d={}'.format(i, j))
    fig.suptitle('GConv kernels')
# convW_visual(temp)

def GConv_visual(model):
    kernels = model.GC_Block3.GConv.pack_kernel(pack_Gab=False)
    n = len(kernels)
    k_size = kernels[0].shape[-1]
    d = kernels[0].shape[0]
    fig, axs = plt.subplots(n, d+1)
    fig.set_size_inches(2.5*(d+1), 2.7*n, forward=True)
    for i in range(n):
        axs[i, 0].imshow(to_array(model.GC_Block3.GConv.Gab[i](k_size)[1, 0]))
        axs[i, 0].title.set_text('Gabor(k={})'.format(i))
        for j in range(0, d):
            axs[i, j+1].imshow(to_array(kernels[i][j, 0]))
            axs[i, j+1].title.set_text('k={} d={}'.format(i, j))
    fig.suptitle('GConv kernels')

def predict_test(folder):
    model = torch.load(folder+'/model')
    os.system("rm -r "+folder+"/test")
    test_list = os.listdir('../testset/')
    os.system("mkdir "+folder+"/test")
    for i in test_list:
        print(i)
        img = cv2.imread('../testset/'+i)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = torch.from_numpy(img)
        img = img.transpose(0, 2).transpose(1, 2)
        img = img.view(1, *img.shape).float()
        img, ori_h, ori_w = code_testimg(img)
        img = PIL_Image.fromarray(to_array(uncode_testimg(model(img)[0], ori_h, ori_w)[0, 0]*255))
        img.save(folder+'/test/'+i[:-4]+'.tif')
        
