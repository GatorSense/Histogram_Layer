# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020

@author: jpeeples
"""
from sklearn.manifold import TSNE
from barbar import Bar
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from matplotlib import offsetbox
from Utils.Compute_FDR import Compute_Fisher_Score
import pdb

def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.05, cmap='copper'):
    # scaler = MinMaxScaler(feature_range=(0,255))
    ax = ax or plt.gca()
    
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            # #Rescale images to be 0 to 255
            # for channel in range(0,images.shape[1]):
            #     scaler.fit(images[i,channel])
            #     scaler.fit_transform(images[i,channel])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i],zoom=.2, cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)
            
def Generate_TSNE_visual(dataloaders_dict,model,sub_dir,device,class_names,
                         histogram=True,Separate_TSNE=False):
    
      #TSNE visual of validation data
        #Get labels and outputs
        GT_val = np.array(0)
        indices_train = np.array(0)
        model.eval()
        model.to(device)
        features_extracted = []
        saved_imgs = []
        for idx, (inputs, classes,index)  in enumerate(Bar(dataloaders_dict['train'])):
            images = inputs.to(device)
            labels = classes.to(device, torch.long)
            indices  = index.to(device).cpu().numpy()
            
            GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
            indices_train = np.concatenate((indices_train,indices),axis = None)
            
            features = model(images)
                
            features = torch.flatten(features, start_dim=1)
            
            features = features.cpu().detach().numpy()
            
            features_extracted.append(features)
            saved_imgs.append(images.cpu().permute(0,2,3,1).numpy())
     
  
        features_extracted = np.concatenate(features_extracted,axis=0)
        saved_imgs = np.concatenate(saved_imgs,axis=0)
        
        #Compute FDR scores
        GT_val = GT_val[1:]
        indices_train = indices_train[1:]
        FDR_scores, log_FDR_scores = Compute_Fisher_Score(features_extracted,GT_val)
        np.savetxt((sub_dir+'train_FDR.txt'),FDR_scores,fmt='%.2E')
        np.savetxt((sub_dir+'train_log_FDR.txt'),log_FDR_scores,fmt='%.2f')
        features_embedded = TSNE(n_components=2,verbose=1,init='random',random_state=42).fit_transform(features_extracted)
        num_feats = features_extracted.shape[1]
    
        fig6, ax6 = plt.subplots()
        colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
        for texture in range (0, len(class_names)):
            x = features_embedded[[np.where(GT_val==texture)],0]
            y = features_embedded[[np.where(GT_val==texture)],1]
            
            ax6.scatter(x, y, color = colors[texture,:],label=class_names[texture])
         
        plt.title('TSNE Visualization of Training Data Features')
        plt.legend(class_names)
        
        # box = ax6.get_position()
        # ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        # ax6.legend(loc='upper center',bbox_to_anchor=(.5,-.05),fancybox=True,ncol=8)
        plt.axis('off')
        plt.show()
        fig6.savefig((sub_dir + '1_TSNE_Visual_Test_Data.png'), dpi=fig6.dpi)
        plt.close()
        
        #Plot tSNE with images
        fig9, ax9 = plt.subplots()
        plot_components(features_extracted,features_embedded,thumb_frac=0.1,images=saved_imgs,cmap=None)
        plt.title('TSNE Visualization of Test Data Features with Images')
        plt.grid('off')
        plt.axis('off')
        plt.show()
        fig9.savefig((sub_dir + '2_TSNE_Visual_Test_Data_Images.png'),dpi=fig9.dpi)
        plt.close()
    

        if (Separate_TSNE):
            conv_features_embedded = TSNE(n_components=2,verbose=1).fit_transform(features_extracted[:,:num_feats//2])
            hist_features_embedded = TSNE(n_components=2,verbose=1).fit_transform(features_extracted[:,num_feats//2:])
            
            GT_val = GT_val[1:]
            indices_train = indices_train[1:]
            fig7, ax7 = plt.subplots()
            colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
            for texture in range (0, len(class_names)):
                x = conv_features_embedded[[np.where(GT_val==texture)],0]
                y = conv_features_embedded[[np.where(GT_val==texture)],1]
                
                plt.scatter(x, y, color = colors[texture,:])
             
            plt.title('TSNE Visualization of Training Data Convolution Features')
            plt.legend(class_names)
            plt.show()
            fig7.savefig((sub_dir + '3_TSNE_Visual_Test_Data_Conv_feats.png'), dpi=fig7.dpi)
            plt.close()
            
            fig10, ax10= plt.subplots()
            plot_components(features_extracted,conv_features_embedded,images=saved_imgs)
            # plt.title('TSNE Visualization of Test Data Features with Images')
            plt.grid('off')
            plt.axis('off')
            plt.show()
            fig10.savefig((sub_dir + '4_TSNE_Visual_Validation_Conv_Data_Images.png'))
            plt.close()
            
            fig8, ax8 = plt.subplots()
            colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
            for texture in range (0, len(class_names)):
                x = hist_features_embedded[[np.where(GT_val==texture)],0]
                y = hist_features_embedded[[np.where(GT_val==texture)],1]
                
                plt.scatter(x, y, color = colors[texture,:])
             
            plt.title('TSNE Visualization of Training Data Histogram Features')
            plt.legend(class_names)
            plt.show()
            fig8.savefig((sub_dir + '5_TSNE_Visual_Test_Data_Hist_feats.png'), dpi=fig8.dpi)
            plt.close()
            
            fig11, ax11 = plt.subplots()
            plot_components(features_extracted,hist_features_embedded,images=saved_imgs)
            # plt.title('TSNE Visualization of Test Data Features with Images')
            plt.grid('off')
            plt.axis('off')
            plt.show()
            fig11.savefig((sub_dir + '6_TSNE_Visual_Validation_Hist_Data_Images.png'))
            plt.close()
        
        # del dataloaders_dict,features_embedded
        torch.cuda.empty_cache()
        
        return FDR_scores, log_FDR_scores