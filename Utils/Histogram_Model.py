from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

class HistRes(nn.Module):
    
    def __init__(self,histogram_layer,parallel=True,model_name ='resnet18'):
        
        #inherit nn.module
        super(HistRes,self).__init__()
        self.parallel = parallel
        #Default to use resnet18, otherwise use Resnet50
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            
        else: 
            print('Model not defined')
            
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        self.fc = self.backbone.fc
        self.backbone.fc = torch.nn.Sequential()
        
        
    def forward(self,x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        #Pass through histogram layer and pooling layer
        if(self.parallel):
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
            x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
            x_combine = torch.cat((x_pool,x_hist),dim=1)
            output = self.fc(x_combine)
        else:
            x = torch.flatten(self.histogram_layer(x),start_dim=1)
            output = self.fc(x)
        
        return output