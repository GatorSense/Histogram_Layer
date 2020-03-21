# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""

######## ONLY CHANGE PARAMETERS BELOW ########
#Flag for if results are to be saved out
save_results = True

#Location to store trained models
folder = 'Saved_Models/'

#Flag to use histogram model or baseline, 
histogram = True
entropy = True

#Select dataset
data_selection = 1
Dataset_names = { 1: 'DTD', 2: 'GTOS-mobile', 3: 'MINC_2500'}

#Set number of workers
num_workers = 0

#Flag for feature extraction. False, train whole model. True, only update 
#fully connected and histogram layers parameters
#Flag to use pretrained model from ImageNet or train from scratch
#Flag to add BN to convolutional features
#Scale at which to apply histogram layer
feature_extraction = False
use_pretrained = True
add_bn = True
scale = 5

#Set learning rate for new and pretrained layers
pt_lr = .001
new_lr = .01

#Set momentum for SGD optimizer
alpha = .9

# Parameters of Histogram Layer
padding = 0
numBins = 4

#Apply rotation to test set
rotation = False
degrees = 25

#Reduce dimensionality based on number of output feature maps
out_channels = {"resnet50": 2048, "resnet18": 512}

#Set whether to have the histogram layer inline or parallel (default: parallel)
#Set whether to use sum (unnormalized count) or average pooling (normalized count)
# (default: average pooling)
#Set whether to enforce sum to one constraint across bins (default: True)
parallel = True
normalize_count = True
normalize_bins = True

#Set step_size and decay rate for scheduler
step_size = 10
gamma = .1

# Batch size for training and epochs
batch_size = {'train': 64, 'val': 256, 'test': 256}
num_epochs = 30

#Resize the image before center crop
resize_size = 256
center_size = 256

#Pin memory for dataloader
pin_memory = True


#Output feature map size after histogram layer
feat_map_size = 4

#Set filter size and stride based on scale
if scale == 1:
    stride = [32, 32] 
    in_channels = {"resnet50": 64, "resnet18": 64}
    kernel_size = {"resnet50": [64,64],  "resnet18": [64,64]}
elif scale == 2:
    stride = [16, 16] 
    in_channels = {"resnet50": 256, "resnet18": 64}
    kernel_size = {"resnet50": [32,32],  "resnet18": [32,32]}
elif scale == 3:
    stride = [8, 8] 
    in_channels = {"resnet50": 512, "resnet18": 128}
    kernel_size = {"resnet50": [16,16],  "resnet18": [16,16]}
elif scale == 4: 
    stride = [4, 4] 
    in_channels = {"resnet50": 1024, "resnet18": 256}
    kernel_size = {"resnet50": [8,8],  "resnet18": [8,8]}
else:
    stride = [2, 2] 
    in_channels = {"resnet50": 2048, "resnet18": 512}
    kernel_size = {"resnet50": [4,4],  "resnet18": [4,4]}


######## ONLY CHANGE PARAMETERS ABOVE ########
if feature_extraction:
    mode = 'Feature_Extraction'
else:
    mode = 'Fine_Tuning'

Data_dirs = {'DTD': './Datasets/DTD/', 
             'MINC_2500': './Datasets/minc-2500/',
             'GTOS-mobile': './Datasets/gtos-mobile'}

Model_names = {'DTD': 'resnet50', 
               'MINC_2500': 'resnet50',
               'GTOS-mobile': 'resnet18'}

num_classes = {'DTD': 47, 
               'MINC_2500': 23,
               'GTOS-mobile': 31}

Splits = {'DTD': 10, 
          'MINC_2500': 5,
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
                      'center_size': center_size, 'padding': padding, 
                      'stride': stride, 'kernel_size': kernel_size,
                      'in_channels': in_channels, 'out_channels': out_channels,
                      'normalize_count': normalize_count, 
                      'normalize_bins': normalize_bins,'parallel': parallel,
                      'numBins': numBins,'feat_map_size': feat_map_size,
                      'Model_names': Model_names, 'num_classes': num_classes, 
                      'Splits': Splits, 'feature_extraction': feature_extraction,
                      'hist_model': Hist_model_name, 'use_pretrained': use_pretrained,
                      'add_bn': add_bn, 'pin_memory': pin_memory, 'scale': scale,
                      'degrees': degrees, 'rotation': rotation}