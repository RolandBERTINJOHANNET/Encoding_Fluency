import torch.nn as nn
import torch
import sys
import os
import hiddenlayer as hl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.model import basic_modules
import warnings

#pixelshuffle module :
class PixelShuffler(nn.Module):
    def __init__(self,upscale_factor):
      super().__init__()
      self.upscale_factor = upscale_factor

    def forward(self,x):
        return nn.functional.pixel_shuffle(x, self.upscale_factor)


#--------------------------------------------THE VAE

class Model(nn.Module):
  #this init function is hard to read, but basically it constructs the VAE with a sparsity constraint on the 
  #layer_sparsity_cstraint-th layers (it being a list)
  def __init__(self,model_name,device,layer_sparsity_cstraint=[],attention=[],sparsity_coeff=1.,sparsity_param=0.001):
    super().__init__()
    self.latent_dim = 512
    self.name = model_name
    self.device=torch.device(str(device))
    
    #some of the constraint indices need to be passed to the vgg_features module, some to the ones after
    vgg_cstraints = [index for index in layer_sparsity_cstraint if index<16]#there's 16 convs in vgg19
    other_cstraints = [index-15 for index in layer_sparsity_cstraint if index>=16]#after vgg, first index is 1
    
    #---------------------encoder
    modules = []
    in_channels = self.latent_dim

    modules.append(basic_modules.VGG19_Features(vgg_cstraints,attention, sparsity_param, sparsity_coeff))
    modules.append(basic_modules.MeanStdFeatureMaps(in_channels,self.latent_dim))
    modules.append(basic_modules.Reparametrization(self.device))
    self.encoder = nn.Sequential(*modules)
    
    #now add the constraints
    for layer_idx in other_cstraints:
        if layer_idx>2:
            raise ValueError("you provided an out-of-bounds index for the sparsity constraint location !")
        self.encoder[layer_idx].add_constraint(sparsity_param,sparsity_coeff)

    #-------------decoder
    #we want a 2-fold upscale factor and a 2-fold depth-decrease factor,
    #thus the number of channels must increase 2-fold each time
    upscale_factor = 2
    in_channels=self.latent_dim
    modules = []
    nb_filters = [self.latent_dim,512, 256, 128, 64]#this is stupidly just used for its length
    out_chan=None#the first iteration sets its value
    for i in range(len(nb_filters)*2):
        if i%2==0:#only downsample once out of two layers
            out_chan = int(in_channels * upscale_factor)#reduced by 2 each iteration
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,out_chan,3,1,padding=1),
                    nn.BatchNorm2d(out_chan),
                    PixelShuffler(upscale_factor=upscale_factor),
                    nn.LeakyReLU()
                    )
                )
            in_channels=int(out_chan/(upscale_factor**2))#the pixelshuffler divides by the square of the number of chans
        else:
            modules.append(
                basic_modules.ResBlock(in_channels)
                )
    
    #add a final layer with a sigmoid,no pixelshuffler and no batch norm
    out_chan = int(in_channels * upscale_factor)#reduced by 2 each iteration
    modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 3, 3,1,1,padding_mode="reflect"),
                nn.Sigmoid())
            )
    self.decoder = nn.Sequential(*modules)

    #get layer names for name-driven activation access by looping over layer names
    self.layers = self.encoder[0].layers
    self.attention_layers = self.encoder[0].attention_layers
    self.layers+=[self.encoder[1].name+("_mu"),self.encoder[1].name+("_std"),self.encoder[2].name]
    print("\n\nlayers : ",self.layers)
    print("\n\nattention_layers : ",self.attention_layers)
    
#----------------------------------------------------------------------------forward
  def forward(self,x):
    #there's just 3 encoder groups
    x,kl,att = self.encoder[0](x)#convs
    x,kl_temp = self.encoder[1](x)#mean-std
    kl+=kl_temp
    x,kl_temp = self.encoder[2](x)#reparametrization
    kl+=kl_temp
    #loop over decoder layers
    for module in self.decoder:
        x = module(x)
    return x,kl,att
    
#----------------------------------------------------------------------------misc
  def get_activations(self,x,layer_name):
      #feature extractor has 16 layers so it is treated separately
      activations_convs = self.encoder[0].get_activations(x,layer_name)
      if activations_convs is not None:
        return activations_convs
      else:#if the layer isn't one of the convs, just feedforward
        x = self.encoder[0](x)[0]
      #try meanstdfeaturemaps
      if "MeanStdFeatureMaps" in layer_name:
          return self.encoder[1].get_activations(x,"mu" if "mu" in layer_name else "sigma")
      else:
          x = self.encoder[1](x)[0]
      #try reparametrization
      if "Reparametrization" in layer_name:
          return self.encoder[2].get_activations(x)
          
 
      raise ValueError("\n!!!!\n!!!!!!\n      name given to model.get_activations ("+str(layer_name)+") doesn't exist !\n")
      return None
  
  def encode(self,x,sample=False):
    for layer in self.encoder[:-1]:
        x,_ = layer(x)
    if sample:#go through the reparametrization if the user wants it
        return self.encoder[-1](x)[0]
    else:
        return x[:,:int(x.shape[1]/2)]#return just the means

  def decode(self,x):
    return self.decoder(x)


  def draw_model(self, filename):
    # Create a hiddenlayer graph from the model
    graph = hl.build_graph(self, torch.zeros([1, 3, 224, 224]).to(self.device))

    # Save the graph to a file
    graph.save(filename)
