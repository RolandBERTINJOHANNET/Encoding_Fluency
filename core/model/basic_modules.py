"""
===========================================
Modules used in building the main model
===========================================

This module provides classes for building a neural network with optional sparsity constraints. It is designed to work with a pretrained VGG19 model, but can be adapted to other models as well.

Classes
-------
**NoConstraint**
    A module that applies a ReLU activation function to its input. (The encoder is built from **ConvBlock**, **MeanStdFeatureMaps** and **Reparametrization** layers. All of them have a **add_constraint** function that switches their **NoConstraint** to **SparsityConstraint**)

**SparsityConstraint**
    A module that applies a sigmoid activation function to its input, then computes the KL divergence between the mean activation and a constant distribution. This is used to encourage sparsity in the activations.

**ResBlock**
    A standard residual block, which consists of two convolutional layers with batch normalization and LeakyReLU activation, followed by an addition operation that adds the input to the output of the convolutions.

**ConvBlock**
    A module that wraps a convolutional layer with a ReLU activation function and an optional sparsity constraint. The sparsity constraint can be added using the `add_constraint` method.

**VGG19_Features**
    A module that extracts non-pretrained VGG19 feature extractor, organizes it in a module where you can apply the sparsity constraints anywhere as well as add attention layers.

**MeanStdFeatureMaps**
    A module which generates a set of featuremaps for means and one for stds. It must be followed by a Reparametrization module.

**Reparametrization**
    A module which performs re-parametrization (sampling from a gaussian, then x sig + mu). The module right before must be a MeanStdFeatureMaps() module.
"""
import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torchvision
from core.model.attention_modules import Constrained_CBAM


class NoConstraint(nn.Module):
    """
    A PyTorch module that applies a ReLU activation function to its input. 
    This class is used as a placeholder when no sparsity constraint is desired.

    Attributes
    ----------
    None

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float]
        Applies the ReLU activation function to the input tensor and returns the result along with 0.0.
    """
    def __init__(self):
      super().__init__()

    def forward(self,x):
        x = nn.ReLU()(x)#apply sigmoid to limit activations to 0,1
        return x,0.
    
class SparsityConstraint(nn.Module):
    """
    A PyTorch module that applies a sigmoid activation function to its input, 
    then computes the KL divergence between the mean activation and a constant distribution. 
    This is used to encourage sparsity in the activations.

    Attributes
    ----------
    constraint_param: float
        The parameter for the constant distribution in the KL divergence calculation.
    scaling_param: float
        The scaling factor for the KL divergence.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float]
        Applies the sigmoid activation function to the input tensor, computes the KL divergence, 
        and returns both the activated tensor and the scaled KL divergence.
    """
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
    """
    A standard residual block, which consists of two convolutional layers with batch normalization 
    and LeakyReLU activation, followed by an addition operation that adds the input to the output 
    of the convolutions.

    Attributes
    ----------
    operations: torch.nn.Sequential
        A sequence of operations that define the residual block.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the operations of the residual block to the input tensor and returns the result.
    """
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


class ConvBlock(nn.Module):
    """
    A PyTorch module that wraps a convolutional layer with a ReLU activation function and an optional 
    sparsity constraint. The sparsity constraint can be added using the `add_constraint` method.

    Attributes
    ----------
    name: str
        The name of the ConvBlock, which includes the index and the activation function used.
    conv: torch.nn.Conv2d
        The convolutional layer wrapped by the ConvBlock.
    constraint: torch.nn.Module
        The constraint applied to the output of the convolutional layer. This is initially a NoConstraint instance, 
        but can be replaced with a SparsityConstraint instance using the `add_constraint` method.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float]
        Applies the convolutional layer and the constraint to the input tensor and returns the result.
    add_constraint(constraint_param: float, scaling_param: float) -> None
        Replaces the current constraint with a SparsityConstraint instance using the provided parameters.
    get_activations(x: torch.Tensor) -> torch.Tensor
        Applies the convolutional layer and the constraint to the input tensor and returns the result without gradients.
    """
    def __init__(self,conv2d,idx):
      super().__init__()
      self.name = "ConvBlock("+str(idx)+")" + "_ReLU"
      
      self.conv = conv2d
      self.constraint = NoConstraint()

    def forward(self,x):
        x = self.conv(x)
        return self.constraint(x)
    
    def add_constraint(self,constraint_param,scaling_param):
        self.constraint = SparsityConstraint(constraint_param, scaling_param)
        self.name = self.name.replace("ReLU","Sigmoid")
        
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x):
        with torch.no_grad():
            x = self.conv(x)
            return self.constraint(x)[0]
    
    
class VGG19_Features(nn.Module):
    """
    A PyTorch module that extracts a non-pretrained VGG19 feature extractor, organizes it in a module where 
    you can apply the sparsity constraints anywhere as well as add attention layers.

    Attributes
    ----------
    name: str
        The name of the VGG19_Features module, which includes the activation function used.
    feature_extractor: torch.nn.Sequential
        A sequence of operations that define the feature extraction process.
    layers: List[str]
        A list of names of the ConvBlock layers in the feature extractor.
    attention_layers: List[str]
        A list of names of the Constrained_CBAM layers in the feature extractor.
    all_layers: List[str]
        A list of names of all the ConvBlock and Constrained_CBAM layers in the feature extractor.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]
        Applies the feature extraction process to the input tensor and returns the result, 
        along with the total KL divergence and attention loss.
    get_activations(x: torch.Tensor, layer_name: str) -> torch.Tensor
        Applies the feature extraction process up to the specified layer to the input tensor and returns the result.
    __len__() -> int
        Returns the number of layers in the feature extractor.
    """
    def __init__(self,constraint_locations,attention,constraint_param,scaling_param):
        super().__init__()
        self.name = "VGG19_Feature_extraction" + "_ReLU"
        features = torchvision.models.vgg19(pretrained=False).features
        self.feature_extractor = self.build_feature_extractor(features, constraint_locations, constraint_param, scaling_param)
        self.add_attention_layers(attention, constraint_param, scaling_param)     
        #get layer names for name-driven activation access by looping over layer names  attention_layers
        self.layers = [layer.name for layer in self.feature_extractor if isinstance(layer,ConvBlock)]
        self.attention_layers = [layer.name for layer in self.feature_extractor if isinstance(layer,Constrained_CBAM)]
        self.all_layers = [layer.name for layer in self.feature_extractor if isinstance(layer,Constrained_CBAM) or isinstance(layer,ConvBlock)]
              

    def build_feature_extractor(self, features, constraint_locations, constraint_param, scaling_param):
        convs = self.build_convs(features, constraint_locations, constraint_param, scaling_param)
        return nn.Sequential(
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

    def build_convs(self, features, constraint_locations, constraint_param, scaling_param):
        convs = []
        i = 0
        for module in features:
            if isinstance(module,torch.nn.Conv2d):
                i += 1
                conv_block = ConvBlock(module, i)
                if i - 1 in constraint_locations:
                    conv_block.add_constraint(constraint_param, scaling_param)
                convs.append(conv_block)
        return convs

    def add_attention_layers(self, attention, constraint_param, scaling_param):
        if True in [att>15 or att<=0 for att in attention]:
          raise ValueError("You provided attention indices that are either too high or too low !")
        nb_filters = [64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
        attention.sort()
        attention.reverse()
        ctr=0#counting convolutions, adding attention whenever needed
        idx=0#iterating through the feature extractor, adding 1 to idx whenever we add a constraint
        while idx<len(self.feature_extractor):
            if isinstance(self.feature_extractor[idx],ConvBlock):
                if ctr in attention:
                    self.feature_extractor.insert(idx,Constrained_CBAM(nb_filters[ctr-1],constraint_param, scaling_param, ctr))
                    idx+=1
                ctr+=1
            idx+=1

    def forward(self,x):
        kl=0.
        att=0.
        for layer in self.feature_extractor:
            #get the kl separately if there's a constraint
            if isinstance(layer, ConvBlock):
                x,kl_temp=layer(x)
                kl+=kl_temp
            #same idea for attention layers
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


class MeanStdFeatureMaps(nn.Module):
    """
    A PyTorch module which generates a set of feature maps for means and one for standard deviations. 
    It must be followed by a Reparametrization module.

    Attributes
    ----------
    name: str
        The name of the MeanStdFeatureMaps module, which includes the activation function used.
    conv_means: torch.nn.Conv2d
        The convolutional layer that generates the mean feature maps.
    conv_stds: torch.nn.Conv2d
        The convolutional layer that generates the standard deviation feature maps.
    constraint: torch.nn.Module
        The constraint applied to the output of the convolutional layers. This is initially a NoConstraint instance, 
        but can be replaced with a SparsityConstraint instance using the `add_constraint` method.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float]
        Applies the convolutional layers and the constraint to the input tensor and returns the result.
    add_constraint(constraint_param: float, scaling_param: float) -> None
        Replaces the current constraint with a SparsityConstraint instance using the provided parameters.
    get_activations(x: torch.Tensor, mean_or_std: str) -> torch.Tensor
        Applies the appropriate convolutional layer and the constraint to the input tensor and returns the result without gradients.
    """
    def __init__(self,in_channels,latent_dim):
      super().__init__()
      self.name = "MeanStdFeatureMaps"+ "_ReLU"
      
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
        self.name = self.name.replace("ReLU","Sigmoid")
    
    
    #mean_or_std is necessary even on non-meanstd layers, because of bad design.
    def get_activations(self,x,mean_or_std):
        with torch.no_grad():
            if mean_or_std!="mu" and mean_or_std!="sigma":
                    raise ValueError("\n\n!!!!!!!name problem in the MeanStdFeatureMaps layer of the network : \n"
                          "        you need to specify '_mu' or '_sigma' after 'MeanStdFeatureMaps'")
                    return None
            x = self.conv_stds(x) if mean_or_std=="sigma" else self.conv_means(x)
            return self.constraint(x)[0]


class Reparametrization(nn.Module):
    """
    A PyTorch module which performs re-parametrization (sampling from a Gaussian, then x sig + mu). 
    The module right before must be a MeanStdFeatureMaps() module.

    Attributes
    ----------
    name: str
        The name of the Reparametrization module, which includes the activation function used.
    constraint: torch.nn.Module
        The constraint applied to the output of the re-parametrization. This is initially a NoConstraint instance, 
        but can be replaced with a SparsityConstraint instance using the `add_constraint` method.
    normal: torch.distributions.Normal
        A Normal distribution used for sampling in the re-parametrization.

    Methods
    -------
    forward(x: torch.Tensor) -> Tuple[torch.Tensor, float]
        Performs the re-parametrization on the input tensor and returns the result.
    add_constraint(constraint_param: float, scaling_param: float) -> None
        Replaces the current constraint with a SparsityConstraint instance using the provided parameters.
    get_activations(x: torch.Tensor) -> torch.Tensor
        Performs the re-parametrization on the input tensor and returns the result without gradients.
    """
    def __init__(self,device):
      super().__init__()
      self.name = "Reparametrization" + "_ReLU"
      
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
        self.name = self.name.replace("ReLU","Sigmoid")
        
    def get_activations(self,x):
        with torch.no_grad():
            mean = x[:,:int(x.shape[1]/2)]
            std = x[:,int(x.shape[1]/2):]
            epsilon = self.normal.sample((std.shape[0],std.shape[1],
                                          std.shape[2],std.shape[3]))
            x = mean + epsilon*std
            return self.constraint(x)[0]