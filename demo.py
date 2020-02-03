# -*- coding: utf-8 -*-
"""
Demo for histogram layer networks (HistRes_B)
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

## Local external libraries
from Utils.Network_functions import initialize_model, train_model,test_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Histogram_Parameters import Network_parameters
from Prepare_Data import Prepare_DataLoaders

import pdb

#Name of dataset
Dataset = Network_parameters['Dataset']

#Model(s) to be used
model_name = Network_parameters['Model_names'][Dataset]

#Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']
num_feature_maps = Network_parameters['Channels'][model_name]

#Local area of feature map after histogram layer
feat_map_size = Network_parameters['feat_map_size']

# Detect if we have a GPU available
# device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Location to store trained models
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, Network_parameters['folder'])

print('Starting Experiments...')

for split in range(0, numRuns):
    
    #Keep track of the bins and widths as these values are updated each
    #epoch
    saved_bins = np.zeros((Network_parameters['num_epochs']+1,numBins*int(num_feature_maps/(feat_map_size*numBins))))
    saved_widths = np.zeros((Network_parameters['num_epochs']+1,numBins*int(num_feature_maps/(feat_map_size*numBins))))
    
    histogram_layer = HistogramLayer(int(num_feature_maps/(feat_map_size*numBins)),
                                     Network_parameters['kernel_size'][model_name],
                                     stride=Network_parameters['stride'],num_bins=numBins)
    
    # Initialize the histogram model for this run
    model_ft, input_size = initialize_model(model_name, num_classes,num_feature_maps, 
                                            feature_extract = Network_parameters['feature_extraction'], 
                                            histogram= Network_parameters['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Network_parameters['parallel'], 
                                            use_pretrained=Network_parameters['use_pretrained'])
    # Send the model to GPU if available
    model_ft = model_ft.to(device)
    
    #Print number of trainable parameters
    num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print("Number of parameters: %d" % (num_params))    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training and validation dataloaders
    dataloaders_dict = Prepare_DataLoaders(Network_parameters,split,input_size=input_size)
    
    #Save the initial values for bins and widths of histogram layer
    #Set optimizer for model
    if(Network_parameters['histogram']):
        saved_bins[0,:] = model_ft.histogram_layer[-1].centers.detach().cpu().numpy()
        saved_widths[0,:] = model_ft.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
        optimizer_ft = optim.SGD([
                {'params': model_ft.backbone.conv1.parameters()},
                {'params': model_ft.backbone.bn1.parameters()},
                {'params': model_ft.backbone.layer1.parameters()},
                {'params': model_ft.backbone.layer2.parameters()},
                {'params': model_ft.backbone.layer3.parameters()},
                {'params': model_ft.backbone.layer4.parameters()},                
                {'params': model_ft.histogram_layer.parameters(), 'lr': Network_parameters['new_lr']},
                {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
            ], lr=Network_parameters['pt_lr'], momentum=Network_parameters['momentum'])
    else:
        saved_bins = None
        saved_widths = None
        optimizer_ft = optim.SGD([
                {'params': model_ft.conv1.parameters()},
                {'params': model_ft.bn1.parameters()},
                {'params': model_ft.layer1.parameters()},
                {'params': model_ft.layer2.parameters()},
                {'params': model_ft.layer3.parameters()},
                {'params': model_ft.layer4.parameters()},
                {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
            ], lr=Network_parameters['pt_lr'], momentum = Network_parameters['momentum'])
   
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                          step_size=Network_parameters['step_size'],
                                          gamma= Network_parameters['gamma'])
    
    # Train and evaluate
    train_dict = train_model(
            model_ft, dataloaders_dict, criterion, optimizer_ft, device,
            saved_bins=saved_bins,saved_widths=saved_widths,histogram=Network_parameters['histogram'],
            num_epochs=Network_parameters['num_epochs'],scheduler=scheduler)
    test_dict = test_model(dataloaders_dict['test'],model_ft,device)
    
    # Save results
    if(Network_parameters['save_results']):
        save_results(train_dict,test_dict,split,Network_parameters,num_params)
        del train_dict,test_dict
        torch.cuda.empty_cache()
        
    if(Network_parameters['histogram']):
        print('**********Run ' + str(split + 1) + ' For ' + Network_parameters['hist_model'] + ' Finished**********') 
    else:
        print('**********Run ' + str(split + 1) + ' For GAP_' + model_name + ' Finished**********') 