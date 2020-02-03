# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:19:34 2019

@author: jpeeples
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,fontsize=12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cb = plt.colorbar(im,fraction=0.046,pad=0.04)
    cb.ax.tick_params(labelsize=fontsize)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = fontsize, rotation=45)
    plt.yticks(tick_marks, classes, fontsize = fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_avg_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,fontsize=12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #Compute average CM values
    std_cm = np.int64(np.ceil(np.std(cm,axis=2)))
    cm = np.int64(np.ceil(np.mean(cm,axis=2)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    cb = plt.colorbar(im,fraction=0.046,pad=0.04)
    cb.ax.tick_params(labelsize=fontsize)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = fontsize, rotation=45)
    plt.yticks(tick_marks, classes, fontsize = fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #pdb.set_trace()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        s = str(format(cm[i, j], fmt)) + '±' + str(format(std_cm[i, j], fmt))
        plt.text(j, i, s,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()