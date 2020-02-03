# -*- coding: utf-8 -*-
"""
Created on Mon July 01 16:01:36 2019
GTOS data loader
@author: jpeeples
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import pdb
import torch
import pandas as pd
import string
import numpy as np

class GTOS_data(Dataset):

    def __init__(self, texture, train = True, numset = 1, img_transform = None):  # numset: 0~5
        self.texture = texture
        self.img_transform = img_transform
        self.files = []  # empty list
        self.targets = [] #labels

        #pdb.set_trace()
        imgset_dir = os.path.join(self.texture)
#        #Ignore \u200e
#        imgset_dir = str(imgset_dir.encode('ascii','ignore'))
        #Load class indices
        class_indices = pd.read_csv('./Datasets/GTOS/labels/classInd.txt',sep=' ', names = ['Index','Label'])

        if train:  # train
            #Get training file
            sample_dir = os.path.join('./Datasets/GTOS/labels/trainlist0' + str(numset) + '.txt')
            #Create dataframe
            data = pd.read_csv(sample_dir, sep = '/', header = None,names = ['Class','Images'] )
            #Loop through data frame and get each image
            for img_folder in range(0,len(data)):
                #Set class label 
                str_label = data['Class'][img_folder]
                #Convert label to numerical value
                bool_label = class_indices['Label'] == str_label
                label = int(np.where(bool_label)[0])
                #Select folder and remove class number/space in name
                temp_img = data['Images'][img_folder].rstrip(string.digits)[:-1]
                temp_img_folder = os.path.join(imgset_dir,temp_img)
                for image in os.listdir(temp_img_folder):
                    if(image=='Thumbs.db'):
                        print('Thumb image') 
                    else:
                        img_file = os.path.join(temp_img_folder,image)
                        self.files.append({  # appends the images
                                "img": img_file,
                                "label": label
                            })
                    self.targets.append(label)

        else:  # test
            sample_dir = os.path.join('./Datasets/GTOS/labels/testlist0' + str(numset) + '.txt')
            #Create dataframe
            data = pd.read_csv(sample_dir, sep = '/', header = None, names = ['Class','Images'] )
            #Loop through data frame and get each image
            for img_folder in range(0,len(data)):
                #Set class label 
                str_label = data['Class'][img_folder]
                #Convert label to numerical value
                bool_label = class_indices['Label'] == str_label
                label = int(np.where(bool_label)[0])
                #Select folder and remove class number/space in name
                temp_img = data['Images'][img_folder].rstrip(string.digits)[:-1]
                temp_img_folder = os.path.join(imgset_dir,temp_img)
                for image in os.listdir(temp_img_folder):
                    if(image=='Thumbs.db'):
                        print('Thumb image') 
                    else:
                        img_file = os.path.join(temp_img_folder,image)
                        self.files.append({  # appends the images
                                "img": img_file,
                                "label": label
                            })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label,index
