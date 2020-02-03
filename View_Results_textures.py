# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019

@author: jpeeples
"""

from __future__ import print_function
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from Utils.Confusion_mats import plot_confusion_matrix,plot_avg_confusion_matrix
from sklearn.metrics import classification_report
from torchvision import transforms
import scipy.stats
import pandas as pd
import os
#from Utils.Generate_hist_vid import Generate_hist_vid
from Datasets.DTD_loader import DTD_data
from Datasets.MINC_2500 import MINC_2500_data
from Datasets.GTOS import GTOS_data
from Datasets.GTOS_mobile_single_size import GTOS_mobile_single_data
import pdb
from Texture_information import Class_names,Data_dirs
from sklearn.manifold import TSNE
from barbar import Bar

Dataset_names = {0: 'KTH', 1: 'DTD', 2: 'GTOS', 3: 'MINC_2500', 4: 'GTOS-mobile'}
Dataset = Dataset_names[3]
data_dir = Data_dirs[Dataset]

#Location of experimental results
device = 'cpu'
fig_size = 13
font_size = 22
stream = 1
gray = 0#for gray or combine stream
directory = "R:/Navy/Individual Folders/Joshua P/New_Histogram_Results/Fine_Tuning/"+Dataset+"/Parallel/4_Bin_Histogram_resnet50/"
#directory = "R:/Navy/Individual Folders/Joshua P/CVPR_results/Fine_Tuning/"+Dataset+"/resnet50/"
NumRuns = len(next(os.walk(directory))[1])
histogram = 0
TSNE_visual = 0
network_name = os.path.basename(os.path.dirname(directory))
plot_name = Dataset +' ' + network_name + ' Test Confusion Matrix'
avg_plot_name = Dataset +' ' + network_name + ' Test Average Confusion Matrix'
class_names = Class_names[Dataset]
cm_stack = np.zeros((len(class_names),len(class_names)))
cm_stats = np.zeros((len(class_names),len(class_names),NumRuns))
accuracy = np.zeros(NumRuns)
resize_size = 256
input_size = 224
batch_num = 2

# Parse through files and plot results
for jj in range(1, NumRuns+1):
    sub_dir = directory + 'Run_' + str(jj) + '/'
    #Load files (Python)
    # Training_Error_track = np.load(sub_dir+'Training_Error_track.npy')
    # Training_Accuracy_track = np.load(sub_dir+'Training_Accuracy_track.npy')
    GT = np.load(sub_dir+'GT.npy')
    predictions = np.load(sub_dir+'Predictions.npy')
    Index = np.load(sub_dir+'Index.npy')
    model = torch.load(sub_dir + 'Model.pt')
    device_loc = torch.device(device)
    best_weights = torch.load(sub_dir + 'Best_Weights.pt',map_location=device_loc)
    model.load_state_dict(best_weights)
    model = model.to(device)
    # Validation_Error_track = np.load(sub_dir+'Validation_Error_track.npy')
    # Validation_Accuracy_track = np.load(sub_dir+'Validation_Accuracy_track.npy')
    Best_epoch = np.load(sub_dir+'best_epoch.npy')
 
    # Training data
    data_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms = None


#    # Create training datasets
#    if Dataset == 'DTD':
#        # Create training and validation datasets
#        train_dataset = DTD_data(data_dir, data='train',
#                                           numset = jj + 1,
#                                           img_transform=data_transforms)
#        validation_dataset = DTD_data(data_dir, data = 'val',
#                                           numset = jj + 1,
#                                           img_transform=data_transforms)
#        #Combine training and validation datasets
#        train_dataset = torch.utils.data.ConcatDataset((train_dataset,validation_dataset))
#    elif Dataset == 'GTOS':
#        # Create training, validation, and test datasets
#        train_dataset = GTOS_data(data_dir, train = True,
#                                           numset = jj + 1,
#                                           img_transform=data_transforms)
#        
#    elif Dataset == 'MINC':
#        # Create training, validation, and test datasets
#        train_dataset = MINC_2500_data(data_dir, data='test',
#                                           numset = jj + 1,
#                                           img_transform=data_transforms)
#    else:
#        train_dataset = GTOS_mobile_single_data(data_dir,train=False, img_transform=data_transforms)    
#    # Create training and validation dataloaders
#    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_num, shuffle=False, num_workers=0)
#    
#    
# 
#    if (TSNE_visual):
#        #TSNE visual of data
#        #Get labels and outputs
#        GT_train = np.array(0)
#        indices_train = np.array(0)
#        model.eval()
##        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
##        feature_extractor = feature_extractor.to(device)
#        features_embedded = []
#        
#    #    pdb.set_trace()
#        loop_count = 0
#        for inputs, classes,index  in dataloader:
#            images = inputs.to(device)
#            labels = classes.to(device, torch.long)
#            indices  = index.to(device).cpu().numpy()
#            
#            GT_train = np.concatenate((GT_train, labels.cpu().numpy()),axis = None)
#            indices_train = np.concatenate((indices_train,indices),axis = None)
#            
#            #Extract features from images
#            #Define forward, code is not currently going through
#            features = model.model.conv1(images)
#            features = model.model.bn1(features)
#            features = model.model.relu(features)
#            features = model.model.maxpool(features)
#            features = model.model.layer1(features)
#            features = model.model.layer2(features)
#            features = model.model.layer3(features)
#            features = model.model.layer4(features)
##            features = feature_extractor(images)
#            
#            features = torch.flatten(features, 1)
#            
#            features = features.cpu().detach().numpy()
#            
#            features_embedded.append(TSNE(n_components=2).fit_transform(features))
#            
#            loop_count += 1
#            print('Loop count: {}/{}'.format(loop_count,len(dataloader)))
#    
#        #Graph TSNE
#    #    colors = {0: "red", 1: "orange", 2: "blue", 3: "purple"}
#        features_embedded = np.asarray(features_embedded)
#        features_embedded = np.concatenate(features_embedded).astype(None)
#        GT_train = GT_train[1:]
#        indices_train = indices_train[1:]
#        fig6, ax6 = plt.subplots()
#        colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
#        for texture in range (0, len(class_names)):
#            x = features_embedded[indices_train[np.where(GT_train==texture)],0]
#            y = features_embedded[indices_train[np.where(GT_train==texture)],1]
#            
#            plt.scatter(x, y, color = colors[texture,:])
#         
#        plt.title('TSNE Visualization of Training Data Features')
#        plt.show()
#        fig6.savefig((sub_dir + 'TSNE_Visual_Training_Data.png'), dpi=fig6.dpi)
#        plt.close()

    
    #Create CM for testing data
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
    plt.plot([Best_epoch], Validation_Error_track[Best_epoch], marker='o', markersize=3, color='red')
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
    fig4 = plt.figure(figsize=(fig_size, fig_size))
    plot_confusion_matrix(cm, classes=class_names, title=plot_name,fontsize=font_size)
    fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
    plt.close()
    cm_stack = cm + cm_stack
    cm_stats[:, :, jj - 1] = cm
    # Get accuracy of each cm
    accuracy[jj - 1] = 100 * sum(np.diagonal(cm)) / sum(sum(cm))
    # Write to text file
    with open((sub_dir + 'Accuracy.txt'), "w") as output:
        output.write(str(accuracy[jj - 1]))

    # Parameters of Histogram Layer
    if (histogram):
        if(stream):
            Bin_centers = model.histogram_layer[-1].centers.detach().cpu().numpy()
            widths = model.histogram_layer[-1].widths.detach().cpu().numpy().reshape(-1)       
        elif(gray):
            Bin_centers = model.histogram_layer.centers.detach().cpu().numpy()
            widths = model.histogram_layer.centers.detach().cpu().numpy()
        else:
            Bin_centers = model.avgpool[-1].centers.detach().cpu().numpy()
            widths = model.avgpool[-1].widths.detach().cpu().numpy()

        # Plot histogram bins and centers with toy data
        fig3 = plt.figure()
        for ii in range(0, len(Bin_centers)):
            toy_data = np.linspace(Bin_centers[ii] - 6 * widths[ii], Bin_centers[ii] + 6 * widths[ii], 300)
            plt.plot(toy_data, abs(scipy.stats.norm.pdf(toy_data, Bin_centers[ii], widths[ii])))

        plt.suptitle('Histogram Learned')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        legend_text = []
#        for histBin in range(0,len(Bin_centers)):
#            legend_text.append('Bin '+ str(histBin+1))
#        plt.legend(legend_text, loc='upper right')
        plt.show()
        fig3.savefig((sub_dir + 'Histogram.png'), dpi=fig3.dpi)
        plt.close()
        
        Generate_hist_vid(sub_dir)

    print('**********Run ' + str(jj) + ' Finished**********')
np.set_printoptions(precision=2)
fig5 = plt.figure(figsize=(fig_size, fig_size))
plot_avg_confusion_matrix(cm_stats, classes=class_names, title=avg_plot_name,fontsize=font_size)
fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
plt.close()
# Write to text file
with open((directory + 'Overall_Accuracy.txt'), "w") as output:
    output.write('Average accuracy: ' + str(np.mean(accuracy)) + ' Std: ' + str(np.std(accuracy)))
print('Mean')
print(np.ceil(np.mean(cm_stats, axis=2)))
print('Std')
print(np.floor(np.std(cm_stats, axis=2)))

plt.close("all")