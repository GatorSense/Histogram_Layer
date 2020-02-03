# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018

@author: jpeeples
"""

import math
import torch
import torch.nn as nn
import numpy as np
import pdb

class HistogramLayer(nn.Module):
    def __init__(self,in_channels,kernel_size,dim=2,num_bins=4,
                 stride=1,padding=0,normalize=True,count_include_pad=False,
                 ceil_mode=False):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.normalize = normalize
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        
        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if dim == 1:
            self.bin_centers_conv = nn.Conv1d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv1d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool1d(self.filt_dim,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        
        # Image Data
        elif dim == 2:
            self.bin_centers_conv = nn.Conv2d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv2d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool2d(self.kernel_size,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
        
        # Spatial/Temporal or Volumetric Data
        elif dim == 3:
            self.bin_centers_conv = nn.Conv3d(self.in_channels,self.numBins*self.in_channels,1,
                                            groups=self.in_channels,bias=True)
            self.bin_centers_conv.weight.data.fill_(1)
            self.bin_centers_conv.weight.requires_grad = False
            self.bin_widths_conv = nn.Conv3d(self.numBins*self.in_channels,
                                             self.numBins*self.in_channels,1,
                                             groups=self.numBins*self.in_channels,
                                             bias=False)
            self.hist_pool = nn.AvgPool3d(self.filt_dim,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
            
        else:
            print('Invalid dimension for histogram layer')
        
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        #Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)
        
        #Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)
        
        #Pass through radial basis function
        xx = torch.exp(-(xx**2))
        
        #Get localized histogram output, if normalize, average count
        if(self.normalize):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size))*self.hist_pool(xx)
        
        return xx
    