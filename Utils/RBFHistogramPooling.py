# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018
Generate histogram layer
@author: jpeeples
"""

import torch
import torch.nn as nn
import numpy as np

class HistogramLayer(nn.Module):
    def __init__(self,in_channels,kernel_size,dim=2,num_bins=4,
                 stride=1,padding=0,normalize_count=True,normalize_bins = True,
                 count_include_pad=False,
                 ceil_mode=False):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        
        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
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
        elif self.dim == 2:
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
        elif self.dim == 3:
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
            raise RuntimeError('Invalid dimension for histogram layer')
        
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window

        #Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)
        
        #Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)
        
        #Pass through radial basis function
        xx = torch.exp(-(xx**2))
        
        #Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if(self.normalize_bins):
            xx = self.constrain_bins(xx)
        
        #Get localized histogram output, if normalize, average count
        if(self.normalize_count):
            xx = self.hist_pool(xx)
        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size))*self.hist_pool(xx)

        return xx
    
    
    def constrain_bins(self,xx):
        #Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n,c,l = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Image Data
        elif self.dim == 2:
            n,c,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n,c,d,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins,d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum   
            
        else:
            raise RuntimeError('Invalid dimension for histogram layer')
         
        return xx
        
        
        
        
        
    