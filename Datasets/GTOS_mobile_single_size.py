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


class GTOS_mobile_single_data(Dataset):

    def __init__(self, texture_dir, train = True,image_size = 256, img_transform = None):  # numset: 0~5
        self.texture_dir = texture_dir
        self.img_transform = img_transform
        self.files = []  # empty list
        self.targets = [] #labels

        #pdb.set_trace()
        imgset_dir = os.path.join(self.texture_dir)

        if train:  # train
            #Get training file
            sample_dir = os.path.join(imgset_dir,'train')
            class_names = sorted(os.listdir(sample_dir))
            label = 0
            #Loop through data frame and get each image
            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    #Check for correct size image
                    if image.startswith(str(image_size)):
                        if(image=='Thumbs.db'):
                            print('Thumb image') 
                        else:
                            img_file = os.path.join(temp_img_folder,image)
                            self.files.append({  # appends the images
                                    "img": img_file,
                                    "label": label
                                })
                            self.targets.append(label)
                label +=1

        else:  # test
            sample_dir = os.path.join(imgset_dir,'test')
            class_names = sorted(os.listdir(sample_dir))
            label = 0
            #Loop through data frame and get each image
            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
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
                label +=1

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

if __name__ == '__main__':
    path = 'gtos-mobile'
#    train = GTOS_mobile_single_data(path)
    test = GTOS_mobile_single_data(path,train=False)