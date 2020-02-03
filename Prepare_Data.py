# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np

## PyTorch dependencies
import torch
from torchvision import transforms

## Local external libraries
from Datasets.DTD_loader import DTD_data
from Datasets.MINC_2500 import MINC_2500_data
from Datasets.GTOS import GTOS_data
from Datasets.GTOS_mobile_single_size import GTOS_mobile_single_data


def Prepare_DataLoaders(Network_parameters, split,input_size=224):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(Network_parameters['resize_size']),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
        # Create training and test datasets
    if Dataset=='DTD':
        train_dataset = DTD_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        validation_dataset = DTD_data(data_dir, data = 'val',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        
        test_dataset = DTD_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['test'])     
        #Combine training and test datasets
        train_dataset = torch.utils.data.ConcatDataset((train_dataset,validation_dataset))  
        
    elif Dataset == 'MINC_2500':
        train_dataset = MINC_2500_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        
        test_dataset = MINC_2500_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['test'])
        
    elif Dataset == 'GTOS':
        train_dataset = GTOS_data(data_dir, train = True,
                                       numset = split + 1,
                                       img_transform=data_transforms['train'])  
        test_dataset = GTOS_data(data_dir, train = False,
                                       numset = split + 1,
                                       img_transform=data_transforms['test'])
    
    else:
        # Create training and test datasets
        train_dataset = GTOS_mobile_single_data(data_dir, train = True,
                                           image_size=Network_parameters['resize_size'],
                                           img_transform=data_transforms['train'])  
        test_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           image_size=Network_parameters['resize_size'],
                                           img_transform=data_transforms['test'])

        
    image_datasets = {'train': train_dataset, 'test': test_dataset}
        

    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=Network_parameters['batch_size'][x], 
                                                       shuffle=True, 
                                                       num_workers=Network_parameters['num_workers']) for x in ['train', 'test']}
    
    return dataloaders_dict