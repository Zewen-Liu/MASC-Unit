#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
from torch.autograd import Variable
from copy import deepcopy
from scipy import signal
import PIL.Image as PIL_Image
import time


class MAC_Unit(nn.Module):
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
        self.conv_w = nn.ParameterList([])

        if mode in ['cosine', 'diffCosine']:
            self.mac_b = nn.ParameterList([])
            self.b2 = nn.ParameterList([])
        self.exp_degree = nn.ParameterList([])
        for i in range(parallel_num):
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
            
        self.in_groups = in_groups

        self.bn = nn.BatchNorm2d(self.angle_num)
        self.padding = self.k_size//2
        # Used for correcting the filter order in direction extention
        self.idx = torch.LongTensor([i for i in range(self.conv_w[0].size(0)-1, -1, -1)])
        # Used for direction channel pattern searching
        self.optimal_pattern = self.get_optimal_pattern()
        
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

        all_kernels = []
        for i in range(self.parallel_num):
            new_conv_w = self.conv_w[i]
            '''
            Do not centrialised kernel, otherwise break the balance, as the kernel 
            of different direction have different positive-negative distribution.
            '''
            new_kernel = self.norm_kernel(new_conv_w*self.Gab[i](scale = self.k_size)[1:forthPi:base_stride], False)
            new_kernel = torch.cat((new_kernel, new_kernel.transpose(2,3).index_select(0, self.idx)), dim=0) 
            new_kernel = torch.cat((new_kernel, torch.rot90(new_kernel, 1, dims=(2, 3))), dim=0)
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
            temp = torch.cat((avg_response, avg_response), dim=1)
            for i in range(self.angle_num):
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
            mask = torch.ones((self.parallel_num, self.angle_num, self.angle_num)) - torch.eye(self.angle_num).unsqueeze(0)
            mask = self.get_p(self.original_pattern, dim=2)[:, :, :, 0, 0]*mask
            return torch.exp((self.get_p((self.get_optimal_pattern(False)+1)/2.01, dim=2)[:, :, :, 0, 0]*mask).sum()/self.parallel_num)
        elif mode == 3:
            return self.get_p(torch.exp(self.get_optimal_pattern(False)+1), dim=2), self.get_p(torch.exp(self.original_pattern*2.01), dim=2)
        elif mode == 4:
            threshold = 2
            
            base_num = int((self.angle_num/2+2)/4)
            forthPi = int(self.angle_num/4)
            halfPi = int(self.angle_num/2)
            base_stride = int(forthPi/base_num)
            
            
            rotational_cost = []
            for i in range(self.k_size**2):
                temp = []
                for j in range(self.parallel_num):
                    temp.append(self.norm_kernel(self.conv_w[j])*self.norm_kernel(Generator_free(self.angle_num, self.k_size)(scale = self.k_size)[1+i%halfPi:forthPi+i%halfPi:base_stride].clone()))
                rotational_cost.append((torch.stack(temp).sum(dim=(-3, -2, -1)).std(1) - threshold).exp() - np.exp(-threshold))
            rotational_cost = torch.cat(rotational_cost, dim=0)
            return rotational_cost

    def update_optimal_pattern(self):
        '''
        Update the optimal_pattern parameter in a GC_node object.
        '''
        r = 0.99
        self.optimal_pattern = self.get_optimal_pattern()*(1-r)+self.optimal_pattern*r
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
        
        # Check the shape of pattern vector
        if optimal_pattern_tmp.shape != (self.parallel_num, self.angle_num, self.angle_num, 1, 1):
            raise ValueError('Found pattern vector in wrong shape.')
            
        if averageM == True:
            temp = torch.cat((optimal_pattern_tmp, optimal_pattern_tmp), dim=2)
            avg_response = temp[:, 0, :self.angle_num]
            for i in range(1, self.angle_num):
                avg_response = avg_response + temp[:, i, i:self.angle_num+i]
            avg_response = avg_response/self.angle_num
            temp = torch.cat((avg_response, avg_response), dim=1)
            for i in range(self.angle_num):
                optimal_pattern_tmp[:, i] = temp[:, self.angle_num-i:2*self.angle_num-i]
        optimal_pattern_tmp = optimal_pattern_tmp.detach()
        
        
        return optimal_pattern_tmp

        
    def soft_max_marginal(self, xbs):
        '''
        Prenonmalised softmax but with non-normalised output.
        '''
        xbs = torch.exp(xbs)
        return xbs.sum(1, keepdim=True)
    
    def diff_max_marginal(self, xbs, l2_norm=True):
        '''
        Prenonmalised diff softmax but with non-normalised output.
        '''
        scale = -1/(self.angle_num)
        feature_map, index_map = xbs.max(1, keepdim=True)
        temp = (xbs!=feature_map)*scale + (xbs==feature_map)*3
        if l2_norm == True:
            xbs = torch.exp(xbs) * temp
        else:
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
            x = torch.exp(self.exp_degree[parallel_i]*x)
            m = torch.exp(self.exp_degree[parallel_i]*self.optimal_pattern[parallel_i])
            m = m - m.mean(dim=1, keepdim=True)*0.75
            norm_m = torch.norm(m, dim=1, keepdim=True)
            x_normed = x/torch.mean(norm_m)
            m_normed = m/norm_m
        else:
            m = self.optimal_pattern[parallel_i]
            m = m - m.mean(dim=1, keepdim=True)*0.75
            norm_m = torch.norm(m, dim=1, keepdim=True)
            x_normed = x/torch.mean(norm_m)
            m_normed = m/norm_m

        temp = F.conv3d(input=x_normed.unsqueeze(1), weight=m_normed.unsqueeze(1))[:, :, 0]
        max_map, index_map = self.mono_max(temp, dim=1, keepdim=True)
        
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
            x = torch.exp(self.exp_degree[parallel_i]*x)
            m = torch.exp(self.exp_degree[parallel_i]*self.optimal_pattern[parallel_i])
            norm_m = torch.norm(m, dim=1, keepdim=True)
            x_normed = x/torch.mean(norm_m)
            m_normed = m/norm_m
        else:
            m = self.optimal_pattern[parallel_i]
            norm_m = torch.norm(m, dim=1, keepdim=True)
            x_normed = x/torch.mean(norm_m)
            m_normed = m/norm_m
        
        temp = F.conv3d(input=x_normed.unsqueeze(1), weight=m_normed.unsqueeze(1))[:, :, 0]
        max_map, index_map = self.mono_max(temp, dim=1, keepdim=True)
        
        return max_map - torch.exp(self.b2[parallel_i]), index_map
    
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
        x = torch.gather(x, 1, temp)
        return self.reorder_combine[parallel_i](x), index_map
        
    def stretch_message_search(self, x, parallel_i):
        '''
        sqrt(sum((r - v)^2)l)
        '''
        c = torch.sqrt(torch.tensor(self.angle_num/2))
        x = x.unsqueeze(2).repeat(1, 1, self.angle_num, 1, 1)
        x = c - torch.norm(x - self.optimal_pattern[parallel_i].unsqueeze(0), dim=2, keepdim=False)
        return self.mono_max(x)
    
    def forward(self, xb, l2_norm=True):
        self.update_optimal_pattern()
        # Must be nature number, The smaller the stonger. Minimum = 1.
        denoise_rate = 1
        temp_parallel = []
        
        kernel = self.pack_kernel()
        if l2_norm == True:
            for j in range(self.parallel_num):
                l2 = self.l2_layer(xb[:, j].unsqueeze(1), self.k_size)
                temp_parallel.append(F.conv2d(xb[:, j].unsqueeze(1), kernel[j], bias=None, stride=1, padding=int(self.padding), groups=self.in_groups)/(l2+0.0001))
        else:
            for j in range(self.parallel_num):
                temp_parallel.append(F.conv2d(xb[:, j].unsqueeze(1), kernel[j], bias=None, stride=1, padding=int(self.padding), groups=self.in_groups))
        
        if l2_norm==True and np.random.randint(0, 10)<1:
            if temp_parallel[0][0].max() > 1.0+self.mac_b[0]:
                print(temp_parallel[0][0].max())
                raise ValueError("Temp maximum exeed 1 issue.")
                
        data = []
        index_maps = []
        for j in range(self.parallel_num):
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
        return torch.cat(data, dim=1), torch.cat(index_maps, dim=1)



class MASC_Block(nn.Module):
    def __init__(self, in_num, mid_num, out_num, angle, k_size=9, inScaleLevel=0, scaleLevel=(1,2,3,4), parallel_num=1, p2nd_k=3, share=False, identical=True):
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
        self.pre_MAC = nn.Conv2d(self.mid_num, parallel_num, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.resi = nn.Conv2d(self.mid_num, self.mid_num*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.share = share
        if self.share == False:
            self.MAC = MAC_Unit(self.angle, self.k_size, parallel_num=parallel_num, identical=identical)
        self.combine = nn.Conv2d((0*in_num+3*self.mid_num+0*self.out_num+1*self.parallel_num)*1+self.parallel_num, self.out_num, kernel_size=1, stride=1, padding=0)
        
        
        self.bn_bottleNeck = nn.BatchNorm2d(self.mid_num)
        self.bn_gc = nn.BatchNorm2d(self.parallel_num)
        self.bn_resi = nn.BatchNorm2d(self.mid_num*2)
        self.bn_combine = nn.BatchNorm2d(self.out_num)
        
        self.indexFeaturize = indexFeaturize_relative(self.angle, self.parallel_num)
        
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
        return output, output_index
    
    def pattern_loss(self, mode=4):
        '''
        Refer to MAC node pattern_loss.
        Return 0 in MAC-sharing mode. 
        '''
        if self.share == True:
            return 0
        else:
            return self.MAC.pattern_loss(mode).mean()
    
    def forward(self, x, sharing_MAC=None, l2_norm=True):
        if self.share == True and sharing_MAC == None:
            ValueError('In share model, must pass a MAC node into the block.')
        x1 = self.bn_bottleNeck(F.leaky_relu(self.bottleNeck(x)))
        resi_x = self.bn_resi(F.leaky_relu(self.resi(x1)))
        
        temp = self.pre_MAC(x1)
        gc = []
        gc_index = []
        for i in range(len(self.scaleLevel)):
            if self.share == False:
                feature_map, index_maps = self.MAC(F.max_pool2d(F.leaky_relu(temp), kernel_size=self.scaleLevel[i], stride=self.scaleLevel[i]), l2_norm=l2_norm)
                gc.append(F.interpolate(feature_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear'))
                gc_index.append(nn.functional.interpolate(index_maps*1.0, scale_factor=self.scaleLevel[i], mode='nearest'))
            if self.share == True:
                feature_map, index_maps = sharing_MAC(F.max_pool2d(F.leaky_relu(temp), kernel_size=self.scaleLevel[i], stride=self.scaleLevel[i]), l2_norm=l2_norm)
                gc.append(F.interpolate(feature_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear'))
                gc_index.append(nn.functional.interpolate(index_maps*1.0, scale_factor=self.scaleLevel[i], mode='nearest'))
        gc = torch.cat(gc, dim=1)
        gc_index = torch.cat(gc_index, dim=1)
        
        gc, gc_index = self.max_channel(gc, gc_index)
        
        if l2_norm == True:
            gc2 = []
            for i in range(self.parallel_num):
                gc2.append(self.exp_message1(self.l2_layer(temp[:, i].unsqueeze(1)), gc[:, i].unsqueeze(1)))
            gc2 = torch.cat(gc2, dim=1)
        else:
            gc2 = self.exp_message1(self.l2_layer(temp, kernel_size=self.k_size), gc)
        b, c, h, w = gc2.shape
        x2 = torch.cat((x1, resi_x, gc2), dim=1)

        
        '''
        Index featurisation
        '''
        temp = self.indexFeaturize(gc_index, gc, self.scaleLevel)
        x2 = torch.cat((x2, temp), dim=1)
        x2 = self.bn_combine(F.leaky_relu(self.combine(x2)))
        return x2, gc2[:, :1], gc.sum(dim=1, keepdim=True)/self.parallel_num, (gc_index/(self.angle-1))*255
    


class MASC_Model(nn.Module):
    def __init__(self, angle, parallel_num, k_size, identical=True):
        super().__init__()
        self.angle = angle
        self.parallel_num = parallel_num
        
        self.l1 = nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.bn_l1 = nn.BatchNorm2d(4)
        self.l2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn_l2 = nn.BatchNorm2d(8)
        # self.l3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.l3 = MASC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.bn_l3 = nn.BatchNorm2d(8)
        
        self.combine1 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.bn_combine1 = nn.BatchNorm2d(8)
        
        self.combine2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.bn_combine2 = nn.BatchNorm2d(8)
        
        
        self.MASC_Block1 = MASC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.MASC_Block2 = MASC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.MASC_Block3 = MASC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        
        self.p2 = MASC_Block(8, 2, 8, angle=self.angle, k_size=k_size, parallel_num=parallel_num, p2nd_k=3, identical=identical)
        self.bn_p2 = nn.BatchNorm2d(8)
        self.p4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
    
    def pack_pattern_loss(self, mode):
        '''
        Pack regularisation loss term on all non-sharing MAC nodes.
        '''
        if mode == 3:
            MASC_Blocks = [self.l3.pattern_loss(mode), self.MASC_Block1.pattern_loss(mode), self.MASC_Block2.pattern_loss(mode), self.MASC_Block3.pattern_loss(mode), self.p2.pattern_loss(mode)]
            
            op = []
            ori = []
            for i, j in MASC_Blocks:
                op.append(i[:, :, : ,0, 0])
                ori.append(j[:, :, : ,0, 0])
            return torch.stack(op), torch.stack(ori)
        elif mode == 4:
            return torch.mean(torch.stack((self.l3.pattern_loss(mode), 
                                           self.MASC_Block1.pattern_loss(mode),
                                           self.MASC_Block2.pattern_loss(mode),
                                           self.MASC_Block3.pattern_loss(mode),
                                           self.p2.pattern_loss(mode))))

    def forward(self, x, l2_norm=True, return_position=-1):
        x = self.bn_l1(F.relu(self.l1(x)))
        x = self.bn_l2(F.relu(self.l2(x)))
        x0 = self.l3(x, l2_norm=l2_norm)
        x = self.bn_l3(F.relu(x0[0]))
        
        x1 = self.MASC_Block1(x, l2_norm=l2_norm)
        x2 = self.MASC_Block2(x1[0], l2_norm=l2_norm)
        
        
        
        temp = torch.cat((x1[0], x2[0]), dim=1)
        temp = self.bn_combine1(F.leaky_relu(self.combine1(temp)))
        
        x3 = self.MASC_Block3(temp, l2_norm=l2_norm)
        
        temp = torch.cat((x, x3[0]), dim=1)
        temp = self.bn_combine2(F.leaky_relu(self.combine2(temp)))
        
        x4 = self.p2(temp, l2_norm=l2_norm)
        fc = self.bn_p2(F.relu(x4[0]))
        fc = torch.sigmoid(self.p4(fc))
        
        target_map = [torch.sigmoid(x0[2]), torch.sigmoid(x1[2]), torch.sigmoid(x2[2]), torch.sigmoid(x3[2]), torch.sigmoid(x4[2]), fc]
        index_maps = [x1[3], x2[3], x3[3]]

        return target_map[return_position], x1[1], x2[1], x3[1], index_maps


    
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
            self.b1 = nn.Parameter(torch.randn(1).uniform_(-0.1, 0.1))
        else:
            self.b1 = 0
        if b2:
            self.b2 = nn.Parameter(torch.randn(1).uniform_(-0.1, 0.1))
        else:
            self.b2 = 0
    
    def forward(self, x1, x2):
        return torch.sigmoid(x1+self.b1)*2+0.00001).pow(x2+self.b2)


    
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
        b, parallel_num, h, w = index_map.shape
        if parallel_num != self.parallel_num:
            raise ValueError('Parameter parallel number({}) does not match input index map\'s parallel number({})'.format(self.parallel_num, parallel_num))   
        featurised_map = self.conv1(index_map)

        for i in range(parallel_num):
            featurised_map[:, i] = featurised_map[:, i] - index_map[:, i]*self.conv1.weight[i, 0].sum()
        
        featurised_map = self.c*x*featurised_map
        
        if smooth:
            featurised_map = F.conv2d(featurised_map, self.gaussian_kernel, groups=self.angle_num*self.parallel_num, padding=self.gaussian_size//2)
        
        feature_maps = []
        for i in range(len(scaleLevel)):
            temp = F.leaky_relu(self.conv3(F.leaky_relu(self.conv2(F.avg_pool2d(featurised_map, kernel_size=scaleLevel[i], stride=scaleLevel[i])))))
            feature_maps.append(F.interpolate(temp, size=(h, w), mode='bilinear'))
        return torch.stack(feature_maps, dim=1).max(dim=1)[0]
    


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
    steerable_cost_involveEpoch = 25
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
            loss.backward()
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
                cv2.imwrite(folder+"/ep"+str(epoch)+'/out1_ep'+str(epoch)+'.png', to_array(a[0][0,0])*255) 
                np.save(folder + '/log', loss_log)
            loss_test = [loss_func(model(xb, l2_norm)[0], yb) for xb, yb in testset]
            loss_test = np.average(loss_test)
            loss_log.append((loss_test, lambda_loss*model.pack_pattern_loss(loss_mode)))
            if loss_test<loss_recorder:
                model_recorder = deepcopy(model)
            lr_schedular.step(loss_test)
            print('epoch:{}\tloss:{:.5f}\t{}*loss_ext:{:.5f}\ttime:{:.5f}'.format(epoch, loss_test, lambda_loss, to_array(lambda_loss*model.pack_pattern_loss(loss_mode)), time.time()-time1))
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






# Test and Visualisation funtions
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
    pattern = to_array(model.MASC_Block3.MAC.test_optimal_pattern()[:, :, :, 0, 0])
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
    kernels = model.MASC_Block3.MAC.conv_w
    n = len(kernels)
    d, _, _, k_size = kernels[0].shape
    fig, axs = plt.subplots(n, d+1)
    fig.set_size_inches(2.5*(d+1), 2.7*n, forward=True)
    for i in range(n):
        axs[i, 0].imshow(to_array(model.MASC_Block3.MAC.Gab[i](k_size)[1, 0]))
        axs[i, 0].title.set_text('Gabor(k={})'.format(i))
        for j in range(0, d):
            axs[i, j+1].imshow(to_array(kernels[i][j, 0]))
            axs[i, j+1].title.set_text('k={} d={}'.format(i, j))
    fig.suptitle('MAC kernels')
# convW_visual(temp)

def MAC_visual(model):
    kernels = model.MASC_Block3.MAC.pack_kernel(pack_Gab=False)
    n = len(kernels)
    k_size = kernels[0].shape[-1]
    d = kernels[0].shape[0]
    fig, axs = plt.subplots(n, d+1)
    fig.set_size_inches(2.5*(d+1), 2.7*n, forward=True)
    for i in range(n):
        axs[i, 0].imshow(to_array(model.MASC_Block3.MAC.Gab[i](k_size)[1, 0]))
        axs[i, 0].title.set_text('Gabor(k={})'.format(i))
        for j in range(0, d):
            axs[i, j+1].imshow(to_array(kernels[i][j, 0]))
            axs[i, j+1].title.set_text('k={} d={}'.format(i, j))
    fig.suptitle('MAC kernels')

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