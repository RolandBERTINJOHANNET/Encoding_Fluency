#-------each class that needs it has an addConstraint function,
#   to make building a network with constraints here and there easier.
import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
import warnings
from core.model.attention_modules import Constrained_CBAM

#-------no-constraint module : a replacement for the sparsity cstraint when we want none.
#(it's literally just a relu, not leaky because VGG19 is not leaky.)
class NoConstraint(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self,x):
        x = nn.ReLU()(x)#apply sigmoid to limit activations to 0,1
        return x,0.
    
#---------------------sparsity constraint module
#applies a relu then computes kl divergence to a constant distribution.
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

#-------ConvBlock : this takes a pretrained conv2D and wraps it with a relu,
#with the possibility of adding a sparsity constraint.
class ConvBlock(nn.Module):
    def __init__(self,conv2d,idx):
      super().__init__()
      self.name = "ConvBlock("+str(idx)+")"
      
      self.conv = conv2d
      self.constraint = NoConstraint()

    def forward(self,x):
        x = self.conv(x)
        return self.constraint(x)
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        print("\n\nadding a sparsity constraint on layer :",self.name,"\n")
        
    def print_name(self):
        print("layer :",self.name,"\n")
        
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x):
        with torch.no_grad():
            x = self.conv(x)
            return self.constraint(x)[0]
    
#-------vgg featuremaps : extracts pretrained vgg19 features extractors, organizes them in a module
#where you can apply the sparsity constraints anywhere
class VGG19_Features(nn.Module):
    def __init__(self,constraint_locations,attention,constraint_param,scaling_param):
      super().__init__()
      self.name = "VGG19_Feature_extraction"
      features = torchvision.models.vgg19(pretrained=False).features
      
      #extract just the convs from VGG19, wrap into ConvBlocks
      convs = []
      i=0#count the convs to give them names
      for module in features:
          if isinstance(module,torch.nn.Conv2d):
              i+=1
              convs.append(ConvBlock(module,i))
      print("number of convs extracted : ",len(convs))
      #add constraints where desired
      for position in constraint_locations:
          convs[position].print_name()
          convs[position].add_constraint(constraint_param,scaling_param)
      #re-build the feature extractor with the pools etc. (ugly, hardcoded :/)
      self.feature_extractor = nn.Sequential(
          convs[0],convs[1],
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          convs[2],convs[3],
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          convs[4],convs[5],convs[6],convs[7],
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          convs[8],convs[9],convs[10],convs[11],
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
          convs[12],convs[13],convs[14],convs[15],
          nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
      
      
      #finally add the attention modules
      nb_filters = [64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
      attention.sort()
      attention.reverse()
      ctr=0#counting convolutions, adding attention whenever needed
      idx=0#iterating through the feature extractor, adding 1 to idx whenever we add a constraint
      while idx<len(self.feature_extractor):
          if isinstance(self.feature_extractor[idx],ConvBlock):
              if ctr in attention:
                  self.feature_extractor.insert(idx,Constrained_CBAM(nb_filters[ctr-1],constraint_param, scaling_param, idx))
                  idx+=1
              ctr+=1
          idx+=1
      #get layer names for name-driven activation access by looping over layer names  attention_layers
      self.layers = [layer.name for layer in self.feature_extractor if isinstance(layer,ConvBlock)]
      self.attention_layers = [layer.name for layer in self.feature_extractor if isinstance(layer,Constrained_CBAM)]
      

    def forward(self,x):
        kl=0.
        att=0.
        for layer in self.feature_extractor:
            #get the kl separately if there's a constraint
            if isinstance(layer, ConvBlock):
                x,kl_temp=layer(x)
                kl+=kl_temp
            elif isinstance(layer,Constrained_CBAM):
                x,att_temp=layer(x)
                att+=kl_temp
            else:
                x=layer(x)
        return x,kl,att
        
    def get_activations(self,x,layer_name):
        with torch.no_grad():
            for layer in self.feature_extractor:
                isLayer = isinstance(layer,ConvBlock) or isinstance(layer,Constrained_CBAM)
                if isLayer and layer.name == layer_name:
                    x = layer.get_activations(x).detach()
                    return x
                else:#keep feeding forward
                    x = layer(x)[0].detach() if isLayer else layer(x).detach() #maxpool doesn't return a kl so no [0]
            return None
           
    def __len__(self):#ignores the pooling and relu "layers"
      return len(self.feature_extractor)

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
    def get_activations(self,x,mean_or_std):
        with torch.no_grad():
            if mean_or_std!="mu" and mean_or_std!="sigma":
                    print(f"mean_or_std : {mean_or_std}")
                    raise ValueError("\n\n!!!!!!!name problem in the MeanStdFeatureMaps layer of the network : \n"
                          "        you need to specify '_mu' or '_sigma' after 'MeanStdFeatureMaps'")
                    return None
            x = self.conv_stds(x) if mean_or_std=="sigma" else self.conv_means(x)
            return self.constraint(x)[0]

#-------Reparametrization module : a module which performs re-parametrization (sampling from a gaussian, then *sig+mu).
#the module right before must be a MeanStdFeatureMaps() module.
class Reparametrization(nn.Module):
    def __init__(self,device):
      super().__init__()
      self.name = "Reparametrization"
      
      self.constraint = NoConstraint()
      #trick to get sampling on the device : pass the params on the device
      self.normal = torch.distributions.Normal(torch.tensor(0.0).to(device),
                                               torch.tensor(1.0).to(device))

    def forward(self,x):
        mean = x[:,:int(x.shape[1]/2)]
        log_stds = x[:,int(x.shape[1]/2):]
        epsilon = self.normal.sample((log_stds.shape[0],log_stds.shape[1],
                                      log_stds.shape[2],log_stds.shape[3]))
        x = mean + epsilon*log_stds
        return self.constraint(x)
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        print("\n\nadding a sparsity constraint on layer :",self.name,"\n")
        
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x):
        with torch.no_grad():
            print(x.shape)
            mean = x[:,:int(x.shape[1]/2)]
            std = x[:,int(x.shape[1]/2):]
            epsilon = self.normal.sample((std.shape[0],std.shape[1],
                                          std.shape[2],std.shape[3]))
            x = mean + epsilon*std
            print(self.constraint(x)[0].shape)
            return self.constraint(x)[0]