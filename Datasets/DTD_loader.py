# -*- coding: utf-8 -*-
"""
Created on Mon July 01 15:20:43 2019
Describale Texture Dataset (DTD) data loader
@author: tk1221
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import pdb
import torch
import matplotlib.pyplot as plt
import statistics
import numpy as np


class DTD_data(Dataset):

    def __init__(self, texture, data='train', numset=1, img_transform=None):
        self.texture = texture
        self.img_transform = img_transform
        self.files = []

        imgset_dir = os.path.join(self.texture)
        # indexing variable for label
        temp_label = 0
        #  initially the first class label
        if data == 'train':  # train
            sample_dir = os.path.join(
                imgset_dir + 'labels/train' + str(numset) + '.txt')  # check if string includes train
            with open(sample_dir) as g:
                line = g.readline()
                comp_label = line[:line.find('/')].rstrip('\n')

            with open(sample_dir) as f:
                for line in f:
                    img_file = line.rstrip('\n')
                    label = line[:line.find('/')]

                    if (comp_label != label):
                        comp_label = label
                        temp_label += 1

                    img_file = os.path.join(imgset_dir + '/images/' + img_file)

                    self.files.append({
                        "img": img_file,
                        "label": temp_label
                    })
        elif data == 'val':  # val
            sample_dir = os.path.join(
                imgset_dir + 'labels/val' + str(numset) + '.txt')  # check if string includes train
            with open(sample_dir) as g:
                line = g.readline()
                comp_label = line[:line.find('/')].rstrip('\n')

            with open(sample_dir) as f:
                for line in f:
                    img_file = line.rstrip('\n')
                    label = line[:line.find('/')]

                    if (comp_label != label):
                        comp_label = label
                        temp_label += 1

                    img_file = os.path.join(imgset_dir + '/images/' + img_file)

                    self.files.append({
                        "img": img_file,
                        "label": temp_label
                    })
        else:  # test
            sample_dir = os.path.join(
                imgset_dir + 'labels/test' + str(numset) + '.txt')  # check if string includes train
            with open(sample_dir) as g:
                line = g.readline()
                comp_label = line[:line.find('/')].rstrip('\n')

            with open(sample_dir) as f:
                for line in f:
                    img_file = line.rstrip('\n')
                    label = line[:line.find('/')]

                    if (comp_label != label):
                        comp_label = label
                        temp_label += 1

                    img_file = os.path.join(imgset_dir + '/images/' + img_file)

                    self.files.append({
                        "img": img_file,
                        "label": temp_label
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

