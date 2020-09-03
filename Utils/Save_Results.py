# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:39:28 2020
Save results from training/testing model
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os

## PyTorch dependencies
import torch

def save_results(train_dict,test_dict,split,Network_parameters,num_params):
    if(Network_parameters['histogram']):
        if(Network_parameters['parallel']):
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                        + '/' + Network_parameters['Dataset'] + '/' 
                        + Network_parameters['hist_model'] +'_'+ 
                        Network_parameters['histogram_type']
                        +'/Parallel/Run_' + str(split + 1) + '/')
        else:
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                        + '/' + Network_parameters['Dataset'] + '/' 
                        + Network_parameters['hist_model'] +'_'+ 
                         Network_parameters['histogram_type']
                        + '/Inline/Run_' + str(split + 1) + '/')
    #Baseline model
    else:
        filename = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/GAP_' +
                    Network_parameters['Model_names'][Network_parameters['Dataset']] + '/Run_' + str(split + 1) + '/')            
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open((filename + 'Num_parameters.txt'), "w") as output:
        output.write(str(num_params))
    np.save((filename + 'Training_Error_track'), train_dict['train_error_track'])
    np.save((filename + 'Test_Error_track'), train_dict['test_acc_track'])
    torch.save(train_dict['best_model_wts'],(filename+'Best_Weights.pt'))
    np.save((filename + 'Training_Accuracy_track'), train_dict['train_acc_track'])
    np.save((filename + 'Test_Accuracy_track'), train_dict['test_acc_track'])
    np.save((filename + 'best_epoch'), train_dict['best_epoch'])
    if(Network_parameters['histogram']):
        np.save((filename + 'Saved_bins'), train_dict['saved_bins'])
        np.save((filename + 'Saved_widths'), train_dict['saved_widths'])
    np.save((filename + 'GT'), test_dict['GT'])
    np.save((filename + 'Predictions'), test_dict['Predictions'])
    np.save((filename + 'Index'), test_dict['Index'])