#-------each class that needs it has an addConstraint function,
#   to make building a network with constraints here and there easier.
import torch
import torch.nn as nn
from torchvision import transforms as transforms
import os
import sys
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
sys.path.insert(1,"/home/renoult/Bureau/internship_cefe_2023/process/sparse_activations_classifier")

#-------no-constraint module : a replacement for the sparsity cstraint when we want none.
#(it's literally just a leaky relu.)
class NoConstraint(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self,x):
        x = nn.LeakyReLU()(x)#apply sigmoid to limit activations to 0,1
        return x,0.

#---------------------sparsity constraint module
class SparsityConstraint(nn.Module):
    def __init__(self,constraint_param,scaling_param):
      super().__init__()
      self.constraint_param = constraint_param
      self.scaling_param = scaling_param

    def forward(self,x):
        x = nn.Sigmoid()(x)#apply sigmoid to limit activations to 0,1
        
        means = x.mean(dim=0)
        #compute kl div to uniform distribution
        kl = (
            self.constraint_param*torch.log(
                (torch.ones_like(means)*self.constraint_param)/means)
        +(1-self.constraint_param)*torch.log(
                (torch.ones_like(means)*(1-self.constraint_param))/(torch.ones_like(means)-means)
                )
            ).mean()
        return x,kl*self.scaling_param


class ResBlock(nn.Module):
  def __init__(self,nb_chan):
    super().__init__()
    self.operations = nn.Sequential(
        nn.Conv2d(nb_chan,nb_chan,3,1,1,padding_mode="reflect"),
        nn.BatchNorm2d(nb_chan),
        nn.LeakyReLU(),
        nn.Conv2d(nb_chan,nb_chan,1,1,0,padding_mode="reflect"),
        nn.BatchNorm2d(nb_chan),
        nn.LeakyReLU()
    )

  def forward(self,x):
    return self.operations(x)+x


#-------MeanStdFeatureMaps module : a module which generates a set of featuremaps for means and one for stds.
#it must be followed by a Reparametrization module.
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_chan,stride):
      super().__init__()
      self.name = "ConvBlock("+str(in_channels)+","+str(out_chan)+")"
      
      self.conv = nn.Conv2d(in_channels,out_chan,3,2 if stride else 1,1,padding_mode="reflect")
      #self.inorm = nn.InstanceNorm2d(out_chan)
      self.constraint = NoConstraint()

    def forward(self,x):
        x = self.conv(x)
        #x = self.inorm(x)
        return self.constraint(x)
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        print("\n\nadding a sparsity constraint on layer :",self.name,"\n")
        
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x,activation=True,mean_or_std=None):
        x = self.conv(x)
        #x = self.inorm(x)
        return nn.Sigmoid()(x) if activation else x

#-------MeanStdFeatureMaps module : a module which generates a set of featuremaps for means and one for stds.
#it must be followed by a Reparametrization mudule.
class MeanStdFeatureMaps(nn.Module):
    def __init__(self,in_channels,latent_dim):
      super().__init__()
      self.name = "MeanStdFeatureMaps"
      
      #convs to get the two param featuremaps for latent space sample parametrization.
      self.conv_means = nn.Conv2d(in_channels,latent_dim,3,1,1,padding_mode="reflect")
      self.conv_stds = nn.Conv2d(in_channels,latent_dim,3,1,1,padding_mode="reflect")
      self.constraint = NoConstraint()

    def forward(self,x):
        means = self.conv_means(x)
        log_stds = self.conv_stds(x)
        x = torch.hstack([means,log_stds])
        return self.constraint(x)
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        print("\n\nadding a sparsity constraint on layer :",self.name,"\n")
    
    
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x,activation=True,mean_or_std=None):
        if mean_or_std!="mean" and mean_or_std!="std":
                print("\n\n!!!!!!!in the MeanStdFeatureMaps layer of the network : \n"
                      "        you need to specify the mean_or_std ('mean' or 'std')"
                      "parameter when getting activations !\n\n",)
                return None
        x = self.conv_stds(x) if mean_or_std=="std" else self.conv_means(x)
        return nn.Sigmoid()(x) if activation else x

#-------Reparametrization module : a module which performs re-parametrization (sampling from a gaussian, then *sig+mu).
#the module right before must be a MeanStdFeatureMaps() module.
#it computes the gaussian constraint in all cases ; you can add a constant constraint on top with add_constraint
class Reparametrization(nn.Module):
    def __init__(self,device):
      super().__init__()
      self.name = "reparametrization (latent sampling)"
      
      self.constraint = NoConstraint()
      #trick to get sampling on the device : pass the params on the device
      self.normal = torch.distributions.Normal(torch.tensor(0.0).to(device),
                                               torch.tensor(1.0).to(device))

    def forward(self,x):
        mean = x[:,:int(x.shape[1]/2)]
        std = x[:,int(x.shape[1]/2):]
        epsilon = self.normal.sample((std.shape[0],std.shape[1],
                                      std.shape[2],std.shape[3]))
        
        kl = 0.5 * ( (mean **2).mean()+ (std**2).mean() 
                    - torch.log(std**2).mean()-1 )#gaussian constraint
        
        x = mean + epsilon*std
        return self.constraint(x)[0],kl*.05
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        print("\n\nadding a sparsity constraint on layer :",self.name,"\n")
        
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x,activation=True,mean_or_std=None):
        mean = x[:,:int(x.shape[1]/2)]
        std = x[:,int(x.shape[1]/2):]
        epsilon = self.normal.sample((std.shape[0],std.shape[1],
                                      std.shape[2],std.shape[3]))
        x = mean + epsilon*std
        return nn.Sigmoid()(x) if activation else x