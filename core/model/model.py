import torch
import torch.nn as nn
import VGG_vae_modules
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
  def __init__(self,device,layer_sparsity_cstraint=[],attention=[],sparsity_coeff=1.,sparsity_param=0.001):
    super().__init__()
    self.latent_dim = 512
    
    #some of the constraint indices need to be passed to the vgg_features module, some to the ones after
    vgg_cstraints = [index for index in layer_sparsity_cstraint if index<16]#thre's 16 convs in vgg19
    other_cstraints = [index-15 for index in layer_sparsity_cstraint if index>16]#after vgg, first index is 1
    
    #---------------------encoder
    modules = []
    module_names = ["feature_extractor","mean_std","reparametrization"]
    in_channels = self.latent_dim

    modules.append(VGG_vae_modules.VGG19_Features(vgg_cstraints,attention, sparsity_param, sparsity_coeff))
    modules.append(VGG_vae_modules.MeanStdFeatureMaps(in_channels,self.latent_dim))
    modules.append(VGG_vae_modules.Reparametrization(device))
    self.encoder = nn.Sequential(*modules)
    
    #now add the constraints
    for layer_idx in other_cstraints:
        print("layer idx : ",layer_idx)
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
                VGG_vae_modules.ResBlock(in_channels)
                )
    
    #add a final layer with a sigmoid,no pixelshuffler and no batch norm
    out_chan = int(in_channels * upscale_factor)#reduced by 2 each iteration
    modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 3, 3,1,1,padding_mode="reflect"),
                nn.Sigmoid())
            )
    self.decoder = nn.Sequential(*modules)

#----------------------------------------------------------------------------forward
  def forward(self,x):
    #loop over encoder layers
    kl = 0.
    for layer in self.encoder:
        x,kl_temp=layer(x)
        kl+=kl_temp
        #print("layer ",type(layer).__name__,": kl = ",float(kl),"\nminmax of x : ",float(x.min()),",",float(x.max()),"\n")
    for module in self.decoder:
        x = module(x)
    return x,kl
    
#----------------------------------------------------------------------------misc
  def get_activations(self,x,layer_num,mean_or_std=None,activation=False):
      layers_done = 0
      #feature extractor has 16 layers so it is treated separately
      if layer_num<16:#get activations from the right layer in feature extractor
          x,name = self.encoder[0].get_activations(x,layer_num,activation,mean_or_std)
          return x,name
      else:#don't get activations, just feedforward
          layers_done+=1
          x = self.encoder[0](x)[0].detach()
      layer_num-=15#we just went through the 16 vgg convs
      #go through the rest of the decoder
      for layer in self.encoder[1:]:
          if layers_done>=layer_num:
              x = layer.get_activations(x,activation,mean_or_std).detach()
              return x,str(type(layer).__name__)+"_"+(mean_or_std if layer_num==1 else "")
          else:
              layers_done+=1
              x = layer(x)[0].detach()
 
      warnings.warn("\n!!!!\n!!!!!!\n      index given to model.get_activations ("+str(layer_num)+") is higher than the number of layers !\n")
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