# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:04:58 2020
Function to compute FDR score per class
@author: jpeeples
"""
import numpy as np

def Compute_Fisher_Score(features,labels):
    
    #Get index of labels that correspond to each class
    Classes = np.unique(labels)
    
    #Get number of instances of each class for P_i
    Instances = np.zeros(len(Classes))
    
    for i in range(0,len(Classes)):
        Instances[i] = sum(labels==Classes[i])
    
    P_i = Instances/sum(Instances);
    
    #Compute global mean
    global_mean = np.mean(features,axis=0)

    #For each class compute intra and inter class variations
    scores = np.zeros(len(Classes))
    log_scores = np.zeros(len(Classes))
    
    for current_class in range(0,len(Classes)):
        data = features[labels==Classes[current_class],:]
        #Within-class scatter matrix
        S_w = P_i[i]*np.cov(data.T)
        #Between-class scatter matrix
        S_b = P_i[i]*(np.outer((np.mean(data,axis=0)-global_mean),
                      (np.mean(data,axis=0)-global_mean).T))
        
        #Compute the score, compute abs of score, only care about magnitude
        #compute log of scores if too large
        #Using pseudoinverse if singular matrix
        try:
            scores[current_class] = abs((np.matmul(np.linalg.inv(S_w),S_b)).trace())
        except:
            scores[current_class] = abs((np.matmul(np.linalg.pinv(S_w),S_b)).trace())    
        log_scores[current_class] = np.log(scores[current_class])
        
    return scores, log_scores