# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Prepare data for visualization
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

## PyTorch dependencies
import torch
from torchvision import transforms

## Local external libraries
from Datasets.DTD_loader import DTD_data
from Datasets.MINC_2500 import MINC_2500_data
from Datasets.GTOS_mobile_single_size import GTOS_mobile_single_data


def Prepare_DataLoaders(Results_parameters, split,input_size=224):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    if not(Network_parameters['rotation']):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(Network_parameters['degrees']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
        # Create training and test datasets
    if Dataset=='DTD':
        train_dataset = DTD_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
        validation_dataset = DTD_data(data_dir, data = 'val',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
        
        test_dataset = DTD_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])     
        # #Combine training and test datasets
        # train_dataset = torch.utils.data.ConcatDataset((train_dataset,validation_dataset))  
        
    elif Dataset == 'MINC_2500':
        train_dataset = MINC_2500_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
        
        validation_dataset = MINC_2500_data(data_dir, data='val',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
        
        test_dataset = MINC_2500_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
    else:
        # Create training and test datasets
        dataset = GTOS_mobile_single_data(data_dir, train = True,
                                          image_size=Network_parameters['resize_size'],
                                          img_transform=data_transforms['val']) 
        X = np.ones(len(train_dataset))
        train_indices = []
        val_indices = []
    
        kf = KFold(n_splits=splits,shuffle=True,
                   random_state=Network_parameters['random_state'])
        
        for train_index, val_index in kf.split(X):
             train_indices.append(train_index)
             val_indices.append(val_index)
        
        train_dataset = torch.utils.data.Subset(dataset, indices['train'][split])
        validation_dataset = torch.utils.data.Subset(dataset, indices['val'][split])
        test_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['val'])

        
    image_datasets = {'train': train_dataset, 'val': validaton_dataset,
                      'test': test_dataset}
        

        
    #If training dataset is larger than number of images for TSNE, subsample
    if len(image_datasets['train']) > Results_parameters['Num_TSNE_images']:
        indices = np.arange(len(image_datasets['train']))
        y = image_datasets['train'].targets
        #Use stratified split to balance training validation splits, 
        #set random state to be same for each encoding method
        _,_,_,_,_,TSNE_indices = train_test_split(y,y,indices,
                                                             stratify=y,
                                                             test_size = Results_parameters['Num_TSNE_images'],
                                                             random_state=split+1)
        
        # Creating PT data samplers and loaders:
        TSNE_sampler = {'train': SubsetRandomSampler(TSNE_indices), 'val': None,'test': None}
    else:
        TSNE_sampler = {'train': None, 'val': None, 'test': None}
        
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                           batch_size=Results_parameters['batch_size'][x], 
                                                           shuffle=False,
                                                           sampler = TSNE_sampler[x],
                                                           num_workers=Results_parameters['num_workers'],
                                                           pin_memory=Results_parameters['pin_memory']) for x in ['train', 'val','test']}
  
    
    return dataloaders_dict