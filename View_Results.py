# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Texture_information import Class_names
from View_Results_Parameters import Results_parameters
from Utils.Network_functions import initialize_model
from Prepare_Data_Results import Prepare_DataLoaders
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Confusion_mats import plot_confusion_matrix,plot_avg_confusion_matrix

#Location of experimental results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fig_size = Results_parameters['fig_size']
font_size = Results_parameters['font_size']

NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]
plot_name = Results_parameters['Dataset'] + ' Test Confusion Matrix'
avg_plot_name = Results_parameters['Dataset'] + ' Test Average Confusion Matrix'
class_names = Class_names[Results_parameters['Dataset']]
cm_stack = np.zeros((len(class_names),len(class_names)))
cm_stats = np.zeros((len(class_names),len(class_names),NumRuns))
FDR_scores = np.zeros((len(class_names),NumRuns))
log_FDR_scores = np.zeros((len(class_names),NumRuns))
accuracy = np.zeros(NumRuns)
MCC = np.zeros(NumRuns)

#Name of dataset
Dataset = Results_parameters['Dataset']

#Model(s) to be used
model_name = Results_parameters['Model_names'][Dataset]

#Number of classes in dataset
num_classes = Results_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Results_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Results_parameters['numBins']
num_feature_maps = Results_parameters['out_channels'][model_name]

#Local area of feature map after histogram layer
feat_map_size = Results_parameters['feat_map_size']

# Parse through files and plot results
for split in range(0, NumRuns):
    
    #Set directory location for experiments
    if(Results_parameters['histogram']):
        if(Results_parameters['parallel']):
            sub_dir = (Results_parameters['folder'] + Results_parameters['mode'] 
                        + '/' + Results_parameters['Dataset'] + '/' 
                        + Results_parameters['hist_model'] + '/Parallel/Run_' 
                        + str(split + 1) + '/')
        else:
            sub_dir = (Results_parameters['folder'] + Results_parameters['mode'] 
                        + '/' + Results_parameters['Dataset'] + '/' 
                        + Results_parameters['hist_model']  + '/Inline/Run_' 
                        + str(split + 1) + '/')
    #Baseline model
    else:
        sub_dir = (Results_parameters['folder'] + Results_parameters['mode'] 
                    + '/' + Results_parameters['Dataset'] + '/GAP_' +
                    Results_parameters['Model_names'][Results_parameters['Dataset']] 
                    + '/Run_' + str(split + 1) + '/') 
        
    #Load files (Python)
    Training_Error_track = np.load(sub_dir+'Training_Error_track.npy')
    Training_Accuracy_track = np.load(sub_dir+'Training_Accuracy_track.npy',allow_pickle=True)
    GT = np.load(sub_dir+'GT.npy')
    predictions = np.load(sub_dir+'Predictions.npy')
    Index = np.load(sub_dir+'Index.npy')
    
    # #Load model
    histogram_layer = HistogramLayer(int(num_feature_maps/(feat_map_size*numBins)),
                                     Results_parameters['kernel_size'][model_name],
                                     num_bins=numBins,stride=Results_parameters['stride'],
                                     normalize_count=Results_parameters['normalize_count'],
                                     normalize_bins=Results_parameters['normalize_bins'])
    
    # Initialize the histogram model for this run
    model, input_size = initialize_model(model_name, num_classes,
                                            Results_parameters['in_channels'][model_name],
                                            num_feature_maps, 
                                            feature_extract = Results_parameters['feature_extraction'], 
                                            histogram= Results_parameters['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Results_parameters['parallel'], 
                                            use_pretrained=Results_parameters['use_pretrained'],
                                            add_bn = Results_parameters['add_bn'],
                                            scale = Results_parameters['scale'],
                                            feat_map_size=feat_map_size)
    
    device_loc = torch.device(device)
    best_weights = torch.load(sub_dir + 'Best_Weights.pt',map_location=device_loc)
    #If parallelized, need to set change model
    if Results_parameters['Parallelize']:
        model = nn.DataParallel(model)
    model.load_state_dict(best_weights)
    model = model.to(device)
    Validation_Error_track = np.load(sub_dir+'Test_Error_track.npy',allow_pickle=True)
    Validation_Accuracy_track = np.load(sub_dir+'Test_Accuracy_track.npy',allow_pickle=True)
    Best_epoch = np.load(sub_dir+'best_epoch.npy')
 

    if (Results_parameters['TSNE_visual']):
        print("Initializing Datasets and Dataloaders...")
        
        dataloaders_dict = Prepare_DataLoaders(Results_parameters,split,input_size=input_size)
        print('Creating TSNE Visual...')
        
        #Remove fully connected layer
        if Results_parameters['Parallelize']:
            model.module.fc = nn.Sequential()
        else:
            model.fc = nn.Sequential()
        
        #Generate TSNE visual
        FDR_scores[:,split], log_FDR_scores[:,split] = Generate_TSNE_visual(
                             dataloaders_dict,
                             model,sub_dir,device,class_names,
                             histogram=Results_parameters['histogram'],
                             Separate_TSNE=Results_parameters['Separate_TSNE'])
   
    #Create CM for testing data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cm = confusion_matrix(GT,predictions)
    #Create classification report
    report = classification_report(GT,predictions,target_names=class_names,output_dict=True)
    #Convert to dataframe and save as .CSV file
    df = pd.DataFrame(report).transpose()
    #Save to CSV
    df.to_csv((sub_dir+'Classification_Report.csv'))
    
    # visualize results
    fig2 = plt.figure()
    plt.plot(Training_Error_track.T)
    plt.plot(Validation_Error_track)
    # Mark best epoch and validation error
    plt.plot([Best_epoch], Validation_Error_track[Best_epoch].cpu(), marker='o', markersize=3, color='red')
    plt.suptitle('Learning Curve for {} Epochs'.format(len(Training_Error_track)))
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['Training', 'Testing', 'Best Epoch'], loc='upper right')
    plt.show()
    fig2.savefig((sub_dir + 'Learning Curve.png'), dpi=fig2.dpi)
    plt.close()
    
    # visualize results
    fig3 = plt.figure()
    plt.plot(Training_Accuracy_track.T)
    plt.plot(Validation_Accuracy_track.T)
    # Mark best epoch and validation error
    plt.plot([Best_epoch], Validation_Accuracy_track[Best_epoch].cpu().numpy(), marker='o', markersize=3, color='red')
    plt.suptitle('Accuracy for {} Epochs'.format(len(Training_Error_track)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Testing', 'Best Epoch'], loc='upper right')
    plt.show()
    fig3.savefig((sub_dir + 'Accuracy Curve.png'), dpi=fig2.dpi)
    plt.close()

    # Confusion Matrix
    np.set_printoptions(precision=2)
    fig4, ax4= plt.subplots(figsize=(fig_size, fig_size))
    plot_confusion_matrix(cm, classes=class_names, title=plot_name,ax=ax4,
                          fontsize=font_size)
    fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
    plt.close()
    cm_stack = cm + cm_stack
    cm_stats[:, :, split] = cm
    
    # Get accuracy of each cm
    accuracy[split] = 100 * sum(np.diagonal(cm)) / sum(sum(cm))
    # Write to text file
    with open((sub_dir + 'Accuracy.txt'), "w") as output:
        output.write(str(accuracy[split]))
        
    #Compute Matthews correlation coefficient
    MCC[split] = matthews_corrcoef(GT,predictions)
    
    # Write to text file
    with open((sub_dir + 'MCC.txt'), "w") as output:
        output.write(str(MCC[split]))

    print('**********Run ' + str(split+1) + ' Finished**********')


directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'   
np.set_printoptions(precision=2)
fig5, ax5 = plt.subplots(figsize=(fig_size, fig_size))
plot_avg_confusion_matrix(cm_stats, classes=class_names, 
                          title=avg_plot_name,ax=ax5,fontsize=font_size)
fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
plt.close()

# Write to text file
with open((directory + 'Overall_Accuracy.txt'), "w") as output:
    output.write('Average accuracy: ' + str(np.mean(accuracy)) + ' Std: ' + str(np.std(accuracy)))
    
# Write to text file
with open((directory + 'Overall_MCC.txt'), "w") as output:
    output.write('Average MCC: ' + str(np.mean(MCC)) + ' Std: ' + str(np.std(MCC)))
    
# Write to text file
with open((directory + 'training_Overall_FDR.txt'), "w") as output:
    output.write('Average FDR: ' + str(np.mean(FDR_scores,axis=1))
                 + ' Std: ' + str(np.std(FDR_scores,axis=1)))
with open((directory + 'training_Overall_Log_FDR.txt'), "w") as output:
    output.write('Average FDR: ' + str(np.mean(log_FDR_scores,axis=1))
                 + ' Std: ' + str(np.std(log_FDR_scores,axis=1)))
    
#Write list of accuracies and MCC for analysis
np.savetxt((directory+'List_Accuracy.txt'),accuracy.reshape(-1,1),fmt='%.2f')
np.savetxt((directory+'List_MCC.txt'),MCC.reshape(-1,1),fmt='%.2f')
    
np.savetxt((directory+'training_List_FDR_scores.txt'),FDR_scores,fmt='%.2E')
np.savetxt((directory+'training_List_log_FDR_scores.txt'),log_FDR_scores,fmt='%.2f')

plt.close("all")