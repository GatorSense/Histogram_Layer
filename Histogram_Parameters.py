# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019

@author: jpeeples
"""

#Parameters for histogram layer experiments
#Only change parameters in this file before running
#demo.py

######## ONLY CHANGE PARAMETERS BELOW ########
#Flag for if results are to be saved out
save_results = True

#Location to store trained models
folder = 'Saved_Models_Sigmoid/'

#Flag to use histogram model or baseline
histogram =True

#Select dataset
data_selection = 1
Dataset_names = { 1: 'DTD', 2: 'GTOS', 3: 'GTOS-mobile', 4: 'MINC_2500'}
num_workers = 0

#Flag for feature extraction. False, train whole model. True, only update 
#fully connected and histogram layers parameters
#Flag to use pretrained model from ImageNet or train from sratch
feature_extraction = True
use_pretrained = True

#Set learning rate for new and pretrained layers
pt_lr = .001
new_lr = .01

#Set momentum for SGD optimizer
alpha = .9

# Parameters of Histogram Layer
padding = 0
stride = [2, 2] 
kernel_size = {"resnet50": [4,4],  "resnet18": [4,4]}
Channels = {"resnet50": 2048, "resnet18": 512}
numBins = 16

#Set whether to have the histogram layer inline or parallel (default: parallel)
#Set whether to use sum (unnormalized count) or average pooling (normalized count)
# (default: average pooling)
parallel = True
normalize = True

#Set step_size and decay rate for scheduler
step_size = 10
gamma = .1

# Batch size for training and epochs
batch_size = {'train': 64, 'val': 256, 'test': 256}
num_epochs = 30

#Resize the image before random crop
resize_size = 256


#Output feature map size after histogram layer
feat_map_size = 4 

######## ONLY CHANGE PARAMETERS ABOVE ########
if feature_extraction:
    mode = 'Feature_Extraction'
else:
    mode = 'Fine_Tuning'

Data_dirs = {'KTH': './Datasets/KTH-TIPS2-b', 'DTD': './Datasets/DTD/', 
             'GTOS': './Datasets/GTOS/images', 'MINC_2500': './Datasets/minc-2500/',
             'GTOS-mobile': './Datasets/gtos-mobile'}

Model_names = {'DTD': 'resnet50', 'MINC_2500': 'resnet50',
               'GTOS': 'resnet18', 'GTOS-mobile': 'resnet18'}

num_classes = {'DTD': 47, 
             'GTOS': 39, 'MINC_2500': 23,
             'GTOS-mobile': 31}

Splits = {'DTD': 10, 
             'GTOS': 5, 'MINC_2500': 5,
             'GTOS-mobile': 5}

Dataset = Dataset_names[data_selection]
data_dir = Data_dirs[Dataset]

Hist_model_name = 'HistRes_' + str(numBins)

#Return dictionary of parameters
Network_parameters = {'save_results': save_results,'folder': folder, 
                      'histogram': histogram,'Dataset': Dataset, 'data_dir': data_dir,
                      'num_workers': num_workers, 'mode': mode,'new_lr': new_lr, 
                      'pt_lr': pt_lr,'momentum': alpha, 'step_size': step_size,
                      'gamma': gamma, 'batch_size' : batch_size, 
                      'num_epochs': num_epochs, 'resize_size': resize_size, 
                      'padding': padding, 'stride': stride, 'kernel_size': kernel_size,
                      'Channels': Channels,'normalize': normalize,'parallel': parallel,
                      'numBins': numBins,'feat_map_size': feat_map_size,
                      'Model_names': Model_names, 'num_classes': num_classes, 
                      'Splits': Splits, 'feature_extraction': feature_extraction,
                      'hist_model': Hist_model_name, 'use_pretrained': use_pretrained}